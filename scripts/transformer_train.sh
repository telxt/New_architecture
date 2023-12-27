cd ../model_train
for ffn_dim in 4096
do
for head_dim in 128
do
for seed in 46
do
for lr in 8e-4
do
echo $ffn_dim
echo $head_dim
echo $lr
echo $seed
echo '---------'
torchrun --nproc_per_node 4 \
transformer_train.py \
    --train_data_file /root/xtlv/data/datasets/wikitext-103-raw/wiki_train.txt \
    --valid_data_file /root/xtlv/data/datasets/wikitext-103-raw/wiki_valid.txt \
    --test_data_file /root/xtlv/data/datasets/wikitext-103-raw/wiki_test.txt \
    --tokenizer_path /root/pubmodels/transformers/llama-2/llama-2-7b-hf \
    --block_size 1024 \
    --embed_dim 1024 \
    --ffn_dim $ffn_dim \
    --num_heads 8 \
    --head_dim $head_dim \
    --num_layers 24 \
    --dropout 0.1 \
    --do_train \
    --do_eval \
    --seed $seed \
    --learning_rate $lr \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 10 \
    --logging_steps 20 \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --warmup_steps 1000 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-5 \
    --lr_scheduler_type cosine \
    --output_dir ../results/0.37b/trans_new/head_dim_${head_dim}_ffn_dim_${ffn_dim}_lr_${lr} \
    --save_strategy epoch \
    --report_to wandb \
    --run_name trans_lr_${lr} > ../results/0.37b/trans_new/head_dim_${head_dim}_ffn_dim_${ffn_dim}_lr_${lr}.log 2>&1
done
done
done
done
# cd ../model_train
# for ffn_dim in 11520 12288 13056 14592 15360 16128 16896 17664 18432 19200
# do
# echo $ffn_dim
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# torchrun --nproc_per_node 8 \
# transformer_train.py \
#     --train_data_file /root/xtlv/data/datasets/wikitext-103-raw/wiki_train.txt \
#     --valid_data_file /root/xtlv/data/datasets/wikitext-103-raw/wiki_valid.txt \
#     --test_data_file /root/xtlv/data/datasets/wikitext-103-raw/wiki_test.txt \
#     --tokenizer_path /root/pubmodels/transformers/llama-2/llama-2-7b-hf \
#     --block_size 512 \
#     --embed_dim 768 \
#     --ffn_dim $ffn_dim \
#     --num_heads 8 \
#     --head_dim 96 \
#     --num_layers 12 \
#     --dropout 0.1 \
#     --do_train \
#     --do_eval \
#     --learning_rate 2.5e-4 \
#     --per_device_train_batch_size 15 \
#     --per_device_eval_batch_size 15 \
#     --num_train_epochs 4 \
#     --logging_steps 20 \
#     --evaluation_strategy steps \
#     --eval_steps 200 \
#     --warmup_steps 1000 \
#     --lr_scheduler_type cosine \
#     --output_dir /root/xtlv/lxt/New_architecture/results/blocksize512_embeddim768_head8_layers12_batchsize120 \
#     --save_strategy no \
#     --report_to wandb \
#     --run_name ffn_train2 > ../results/blocksize512_embeddim768_head8_layers12_batchsize120/ffn_dim_${ffn_dim}.log 2>&1
# done
# cd ../model_train
# for ffn_dim in 48 24 12 6 3 1
# do
# echo $ffn_dim
# CUDA_VISIBLE_DEVICES=0,1,2 \
# torchrun --nproc_per_node 3 \
# transformer_train.py \
#     --train_data_file /root/xtlv/data/datasets/wikitext-103-raw/wiki_train.txt \
#     --valid_data_file /root/xtlv/data/datasets/wikitext-103-raw/wiki_valid.txt \
#     --test_data_file /root/xtlv/data/datasets/wikitext-103-raw/wiki_test.txt \
#     --tokenizer_path /root/pubmodels/transformers/llama-2/llama-2-7b-hf \
#     --block_size 512 \
#     --embed_dim 768 \
#     --ffn_dim $ffn_dim \
#     --num_heads 8 \
#     --num_layers 12 \
#     --dropout 0.1 \
#     --do_train \
#     --do_eval \
#     --learning_rate 2.5e-4 \
#     --per_device_train_batch_size 40 \
#     --per_device_eval_batch_size 40 \
#     --num_train_epochs 4 \
#     --logging_steps 20 \
#     --evaluation_strategy steps \
#     --eval_steps 200 \
#     --warmup_steps 1000 \
#     --lr_scheduler_type cosine \
#     --output_dir /root/xtlv/lxt/New_architecture/results/blocksize512_embeddim768_head8_layers12_batchsize120 \
#     --save_strategy no \
#     --report_to wandb \
#     --run_name ffn_train2 > ../results/blocksize512_embeddim768_head8_layers12_batchsize120/ffn_dim_${ffn_dim}.log 2>&1
# done
# cd ../model_train
# CUDA_VISIBLE_DEVICES=0 \
# torchrun --nproc_per_node 1 \
# transformer_train.py \
#     --train_data_file /root/xtlv/data/datasets/wikitext-103-raw/wiki_test.txt \
#     --valid_data_file /root/xtlv/data/datasets/wikitext-103-raw/wiki_valid.txt \
#     --test_data_file /root/xtlv/data/datasets/wikitext-103-raw/wiki_test.txt \
#     --tokenizer_path /root/pubmodels/transformers/llama-2/llama-2-7b-hf \
#     --block_size 512 \
#     --embed_dim 768 \
#     --ffn_dim 768 \
#     --num_heads 8 \
#     --num_layers 12 \
#     --dropout 0.1 \
#     --do_train \
#     --do_eval \
#     --learning_rate 2.5e-4 \
#     --per_device_train_batch_size 15 \
#     --per_device_eval_batch_size 15 \
#     --num_train_epochs 4 \
#     --logging_steps 20 \
#     --evaluation_strategy steps \
#     --eval_steps 200 \
#     --warmup_steps 1000 \
#     --lr_scheduler_type cosine \
#     --output_dir /root/xtlv/lxt/New_architecture/results/nouse \
#     --save_strategy no \
#     --report_to none