"""
说明：
position_embedding和gpt2保持一致，直接用了nn.Embedding
参考Bart，Attention和FFN层中的linear有bias
目前的mask只是casual mask，没有实现padding mask
torch的默认初始化方法：Linear是kaiming_uniform，Embedding是normal_（正态分布初始化）, 详见reset_parameters方法
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.query_proj = torch.nn.Linear(embed_dim, num_heads * head_dim)
        self.key_proj = torch.nn.Linear(embed_dim, num_heads * head_dim)
        self.value_proj = torch.nn.Linear(embed_dim, num_heads * head_dim)
        self.out_proj = torch.nn.Linear(num_heads * head_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, input_dim = x.size()
        q = self.query_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        mask = torch.tril(torch.ones(seq_len, seq_len)).to(x.device)
        attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return self.out_proj(attn_output)


class FFNTransLayer(nn.Module):
    def __init__(self, embed_dim, ffn_dim, num_heads, head_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, head_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # attn_output = self.attention(x)
        # x = x + self.dropout1(self.norm1(attn_output))
        # fc_output = self.fc(x)
        # x = x + self.dropout2(self.norm2(fc_output))
        # return x

        attn_output = self.attention(x)
        # post norm
        x = self.norm1(x + self.dropout1(attn_output))
        fc_output = self.fc(x)
        # post norm
        x = self.norm2(x + self.dropout2(fc_output))
        return x



class FFN_Transformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, ffn_dim, num_heads, head_dim, num_layers, seq_len, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        # self.positional_embedding = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        self.positional_embedding = nn.Embedding(seq_len, embed_dim)
        self.transformer_blocks = nn.ModuleList([
            FFNTransLayer(embed_dim, ffn_dim, num_heads, head_dim, dropout) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        # x = self.token_embedding(x) + self.positional_embedding
        x = self.token_embedding(x) + self.positional_embedding(torch.arange(x.shape[1]).to(x.device))
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        x = self.fc(x)
        return x


class FFN_TransForCausalLM(nn.Module):
    def __init__(
        self, 
        vocab_size, 
        embed_dim,
        ffn_dim,
        num_heads, 
        head_dim,
        num_layers, 
        seq_len, 
        dropout=0.1, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.FFN_transformer = FFN_Transformer(vocab_size, embed_dim, ffn_dim, num_heads, head_dim, num_layers, seq_len, dropout)

        self.vocab_size = vocab_size

    def forward(
        self,
        input_ids,
        labels,
        **kwargs
    ):
        logits = self.FFN_transformer(x=input_ids,)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return [loss]
    





# ## old version using torch.nn.MultiheadAttention
# """
# 说明：
# position_embedding和gpt2保持一致，直接用了nn.Embedding
# """

# import torch
# import torch.nn as nn


# class FFNTransLayer(nn.Module):
#     def __init__(self, embed_dim, ffn_dim, num_heads, dropout=0.1):
#         super().__init__()
#         self.attention = nn.MultiheadAttention(embed_dim, num_heads)
#         ## MultiheadAttention的embed_dim是多头注意力的总维度数量，而不是单个头的维度数量
#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.dropout1 = nn.Dropout(dropout)
#         self.fc = nn.Sequential(
#             nn.Linear(embed_dim, ffn_dim),
#             nn.ReLU(),
#             nn.Linear(ffn_dim, embed_dim),
#         )
#         self.norm2 = nn.LayerNorm(embed_dim)
#         self.dropout2 = nn.Dropout(dropout)

#     def forward(self, x, attention_mask):
#         attn_output, _ = self.attention(x, x, x, attn_mask=attention_mask)
#         x = x + self.dropout1(self.norm1(attn_output))
#         fc_output = self.fc(x)
#         x = x + self.dropout2(self.norm2(fc_output))
#         return x


# class FFN_Transformer(nn.Module):
#     def __init__(self, vocab_size, embed_dim, ffn_dim, num_heads, num_layers, seq_len, dropout=0.1):
#         super().__init__()
#         self.token_embedding = nn.Embedding(vocab_size, embed_dim)
#         # self.positional_embedding = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
#         self.positional_embedding = nn.Embedding(seq_len, embed_dim)
#         self.transformer_blocks = nn.ModuleList([
#             FFNTransLayer(embed_dim, ffn_dim, num_heads, dropout) for _ in range(num_layers)
#         ])
#         self.fc = nn.Linear(embed_dim, vocab_size)

#     def forward(self, x, attention_mask):
#         # x = self.token_embedding(x) + self.positional_embedding
#         x = self.token_embedding(x) + self.positional_embedding(torch.arange(x.shape[1]).to(x.device))
#         for transformer_block in self.transformer_blocks:
#             x = transformer_block(x, attention_mask)
#         x = self.fc(x)
#         return x


# class FFN_TransForCausalLM(nn.Module):
#     def __init__(
#         self, 
#         vocab_size, 
#         embed_dim,
#         ffn_dim,
#         num_heads, 
#         num_layers, 
#         seq_len, 
#         dropout=0.1, 
#         **kwargs
#     ):
#         super().__init__(**kwargs)
#         self.FFN_transformer = FFN_Transformer(vocab_size, embed_dim, ffn_dim, num_heads, num_layers, seq_len, dropout)

#         self.vocab_size = vocab_size

#     def forward(
#         self,
#         input_ids,
#         labels,
#         attention_mask,
#         **kwargs
#     ):
#         logits = self.FFN_transformer(x=input_ids, attention_mask=attention_mask)
        
#         loss = None
#         if labels is not None:
#             # Shift so that tokens < n predict n
#             shift_logits = logits[..., :-1, :].contiguous()
#             shift_labels = labels[..., 1:].contiguous()
#             # Flatten the tokens
#             loss_fct = nn.CrossEntropyLoss()
#             shift_logits = shift_logits.view(-1, self.vocab_size)
#             shift_labels = shift_labels.view(-1)
#             # Enable model parallelism
#             shift_labels = shift_labels.to(shift_logits.device)
#             loss = loss_fct(shift_logits, shift_labels)

#         return [loss]