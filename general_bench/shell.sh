# vicuna
python answer_vicuna.py --model_name /root/MODELS/vicuna-7b-v1.3 --model_id vicuna-7b-v1.3 --bench_name mt_bench --ssd_name /root/idea/speculative_decoding/ssd_hand/train/ssd_models/vicuna-7b-v1.3_ssd_3_lr_0.002_dim_1024 --attn

python answer_vicuna.py --model_name /root/MODELS/vicuna-7b-v1.3 --model_id vicuna-7b-v1.3 --bench_name mt_bench --medusa

python answer_vicuna.py --model_name /root/MODELS/vicuna-7b-v1.3 --model_id vicuna-7b-v1.3 --bench_name mt_bench

python speed.py --model_name /root/MODELS/vicuna-7b-v1.3 --model_id vicuna-7b-v1.3 --bench_name mt_bench --attn


# llama2
python answer_llama2.py --model_name /root/MODELS/Llama-2-7b-chat-hf --model_id llama-2-7b-chat --bench_name mt_bench --max_seq_length 4096 --ssd_name /root/idea/speculative_decoding/ssd_hand/train/ssd_models/Llama-2-7b-chat-hf_ssd_3_lr_0.002_dim_1024 --attn

python answer_llama2.py --model_name /root/MODELS/Llama-2-7b-chat-hf --model_id llama-2-7b-chat --bench_name mt_bench --max_seq_length 4096

python speed.py --model_name /root/MODELS/Llama-2-7b-chat-hf --model_id llama-2-7b-chat --bench_name mt_bench --attn