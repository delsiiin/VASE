WANDB_MODE="offline" CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 main.py --model_name_or_path /root/MODELS/vicuna-7b-v1.3 \
    --data_path /root/idea/speculative_decoding/VASE/gen_data/train_data_vicuna7b \
    --bf16 True \
    --output_dir ssd_models/ \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "steps" \
    --eval_steps 1000 \
    --save_strategy "steps" \
    --save_steps 4000 \
    --learning_rate 3e-5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --lazy_preprocess False \
    --top_layers_len 24 \
    --top_k_group 3 \
    --resnet_num 1 \
    --attn_hid_dim 4096  \
    --ssd_decay_coefficient 0.8 \
    --model_type vicuna 


# WANDB_MODE="offline" CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 main.py --model_name_or_path /root/MODELS/vicuna-7b-v1.3 \
#     --data_path /root/idea/speculative_decoding/VASE/gen_data/train_data_vicuna7b \
#     --bf16 True \
#     --output_dir ssd_models/ \
#     --num_train_epochs 2 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 4 \
#     --evaluation_strategy "steps" \
#     --eval_steps 1000 \
#     --save_strategy "steps" \
#     --save_steps 4000 \
#     --learning_rate 2e-3 \
#     --weight_decay 0.0 \
#     --warmup_ratio 0.1 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --lazy_preprocess False \
#     --top_layers_len 24 \
#     --top_k_group 3 \
#     --resnet_num 1 \
#     --attn_hid_dim 1024  \
#     --ssd_decay_coefficient 0.8 \
#     --model_type vicuna 
