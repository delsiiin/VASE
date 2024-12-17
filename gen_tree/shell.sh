python heads_accuracy.py --model_path /root/MODELS/vicuna-7b-v1.3 \
                        --model_name vicuna-7b-v1.3 \
                        --data_path ./alpaca_eval.json\
                        --attn\
                        --ssd_name /root/idea/speculative_decoding/ssd_hand/train/ssd_models/vicuna-7b-v1.3_ssd_3_lr_0.002_dim_1024

python gen_results.py --accuracy-path ./data/vicuna-7b-v1.3-24-3_heads_accuracy.pt --output-path ./data/graph.jpg