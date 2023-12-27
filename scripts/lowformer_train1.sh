cd ../model_train
for attn_lowdim in 256
do
for lr in 8e-4
do
for seed in 27 87
do
echo $attn_lowdim
echo $lr
echo $seed
echo '---------'
torchrun --nproc_per_node 6 \
lowformer_train1.py \
    --train_data_file /root/xtlv/data/datasets/wikitext-103-raw/wiki_train.txt \
    --valid_data_file /root/xtlv/data/datasets/wikitext-103-raw/wiki_valid.txt \
    --test_data_file /root/xtlv/data/datasets/wikitext-103-raw/wiki_test.txt \
    --tokenizer_path /root/pubmodels/transformers/llama-2/llama-2-7b-hf \
    --block_size 1024 \
    --embed_dim 1024 \
    --ffn_dim 4096 \
    --num_heads 8 \
    --head_dim 384 \
    --attn_lowdim $attn_lowdim \
    --ffn_lowdim 0 \
    --num_layers 24 \
    --dropout 0.1 \
    --low_attn yes \
    --low_ffn no \
    --do_train \
    --do_eval \
    --seed $seed \
    --learning_rate $lr \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 12 \
    --num_train_epochs 10 \
    --logging_steps 20 \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --warmup_steps 1000 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-5 \
    --lr_scheduler_type cosine \
    --output_dir ../results/0.37b/low_attn/attn_lowdim_${attn_lowdim}_lr_${lr}_seed_${seed} \
    --save_strategy epoch \
    --report_to none \
    --run_name low37attn_low_attn${attn_lowdim}_lr_${lr} > ../results/0.37b/low_attn/attn_lowdim_${attn_lowdim}_lr_${lr}_seed_${seed}.log 2>&1
done
done
done
# cd ../model_train
# for ffn_lowdim in 64
# do
# for attn_lowdim in 2112
# do
# for lr in 3e-4
# do
# seed=87
# echo $ffn_lowdim
# echo $lr
# echo $seed
# echo '---------'
# CUDA_VISIBLE_DEVICES=2,3 \
# torchrun --nproc_per_node 2 --master_port 8690 \
# lowformer_train1.py \
#     --train_data_file /root/xtlv/data/datasets/wikitext-103-raw/wiki_train.txt \
#     --valid_data_file /root/xtlv/data/datasets/wikitext-103-raw/wiki_valid.txt \
#     --test_data_file /root/xtlv/data/datasets/wikitext-103-raw/wiki_test.txt \
#     --tokenizer_path /root/pubmodels/transformers/llama-2/llama-2-7b-hf \
#     --block_size 512 \
#     --embed_dim 768 \
#     --ffn_dim 36096 \
#     --num_heads 8 \
#     --head_dim 96 \
#     --attn_lowdim $attn_lowdim \
#     --ffn_lowdim $ffn_lowdim \
#     --num_layers 12 \
#     --dropout 0.1 \
#     --low_attn no \
#     --low_ffn yes \
#     --do_train \
#     --do_eval \
#     --seed $seed \
#     --learning_rate $lr \
#     --per_device_train_batch_size 32 \
#     --per_device_eval_batch_size 32 \
#     --num_train_epochs 10 \
#     --logging_steps 20 \
#     --evaluation_strategy steps \
#     --eval_steps 200 \
#     --warmup_steps 1000 \
#     --lr_scheduler_type cosine \
#     --output_dir ../results/lowformer/just_ffn/ffn_lowdim_${ffn_lowdim}_ffn_dim_36096_lr_${lr}_seed_${seed} \
#     --save_strategy epoch \
#     --report_to wandb \
#     --run_name low_ffn${ffn_lowdim}_lr_${lr} > ../results/lowformer/just_ffn/ffn_lowdim_${ffn_lowdim}_ffn_dim_36096_lr_${lr}_seed_${seed}.log 2>&1
# done
# done
# done
