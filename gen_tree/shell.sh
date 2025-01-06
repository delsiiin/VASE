# vicuna
python heads_accuracy.py --model_path /root/MODELS/vicuna-7b-v1.3 \
                        --model_name vicuna-7b-v1.3 \
                        --data_path ./alpaca_eval.json\
                        --attn \
                        --ssd_name /root/idea/speculative_decoding/VASE/train/ssd_models/vicuna-7b-v1.3_ssd_3_lr_0.002_dim_1024

python gen_results.py --accuracy-path ./data/vicuna-7b-v1.3-24-3_heads_accuracy.pt \
                    --output-path ./data/graph.jpg \
                    --max-depth 3 \
                    --num-iterations 24

# llama2
python heads_accuracy.py --model_path /root/MODELS/Llama-2-7b-chat-hf \
                        --model_name llama2-7b-chat \
                        --data_path ./alpaca_eval.json\
                        --attn \
                        --ssd_name /root/idea/speculative_decoding/VASE/train/ssd_models/Llama-2-7b-chat-hf_ssd_3_lr_0.002_dim_1024

python gen_results.py --accuracy-path ./data/llama2-7b-chat-24-3_heads_accuracy.pt \
                    --output-path ./data/graph.jpg \
                    --max-depth 3 \
                    --num-iterations 24