# vicuna 7b
python answer_vicuna.py --model_name /root/MODELS/vicuna-7b-v1.3 --model_id vicuna-7b-v1.3 --bench_name mt_bench --ssd_name /root/idea/speculative_decoding/VASE/train/ssd_models/vicuna-7b-v1.3_ssd_3_lr_0.002_dim_1024 --attn

python answer_vicuna.py --model_name /root/MODELS/vicuna-7b-v1.3 --model_id vicuna-7b-v1.3 --bench_name mt_bench --medusa

python answer_vicuna.py --model_name /root/MODELS/vicuna-7b-v1.3 --model_id vicuna-7b-v1.3 --bench_name mt_bench

python speed.py --model_name /root/MODELS/vicuna-7b-v1.3 --model_id vicuna-7b-v1.3 --bench_name mt_bench --attn


# vicuna 13b
python answer_vicuna.py --model_name /root/MODELS/vicuna-13b-v1.3 --model_id vicuna-13b-v1.3 --bench_name mt_bench --ssd_name /root/idea/speculative_decoding/VASE/train/ssd_models/vicuna-13b-v1.3_ssd_3_lr_0.002_dim_1280 --attn

python answer_vicuna.py --model_name /root/MODELS/vicuna-13b-v1.3 --model_id vicuna-13b-v1.3 --bench_name mt_bench --medusa

python answer_vicuna.py --model_name /root/MODELS/vicuna-13b-v1.3 --model_id vicuna-13b-v1.3 --bench_name mt_bench

python speed.py --model_name /root/MODELS/vicuna-13b-v1.3 --model_id vicuna-13b-v1.3 --bench_name mt_bench --attn


# llama2 7b
python answer_llama2.py --model_name /root/MODELS/Llama-2-7b-chat-hf --model_id llama-2-7b-chat --bench_name mt_bench --max_seq_length 4096 --ssd_name /root/idea/speculative_decoding/VASE/train/ssd_models/Llama-2-7b-chat-hf_ssd_3_lr_0.002_dim_1024 --attn

python answer_llama2.py --model_name /root/MODELS/Llama-2-7b-chat-hf --model_id llama-2-7b-chat --bench_name mt_bench --max_seq_length 4096

python speed.py --model_name /root/MODELS/Llama-2-7b-chat-hf --model_id llama-2-7b-chat --bench_name mt_bench --attn


# llama2 13b
python answer_llama2.py --model_name /root/MODELS/Llama-2-13b-chat-hf --model_id llama-2-13b-chat --bench_name mt_bench --max_seq_length 4096 \
                        --ssd_name /root/idea/speculative_decoding/VASE/train/ssd_models/Llama-2-13b-chat-hf_ssd_3_lr_0.002_dim_1280 --attn \
                        --load_in_8bit

python answer_llama2.py --model_name /root/MODELS/Llama-2-13b-chat-hf --model_id llama-2-13b-chat --bench_name mt_bench --max_seq_length 4096 --load_in_8bit

python speed.py --model_name /root/MODELS/Llama-2-13b-chat-hf --model_id llama-2-13b-chat --bench_name mt_bench --attn


# llama3 8b
python answer_llama3.py --model_name /root/MODELS/Meta-Llama-3-8B-Instruct --model_id llama-3-8b-inst --bench_name mt_bench --max_seq_length 4096 --ssd_name /root/idea/speculative_decoding/VASE/train/ssd_models/Meta-Llama-3-8B-Instruct_ssd_3_lr_0.002_dim_1024 --attn

python answer_llama3.py --model_name /root/MODELS/Meta-Llama-3-8B-Instruct --model_id llama-3-8b-inst --bench_name mt_bench --max_seq_length 4096

python speed.py --model_name /root/MODELS/Meta-Llama-3-8B-Instruct --model_id llama-3-8b-inst --bench_name mt_bench --attn