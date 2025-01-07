# vicuna 7b
python allocation.py --outdir train_data_vicuna7b \
                    --type vicuna \
                    --basemodel /root/MODELS/vicuna-7b-v1.3 \
                    --data /root/idea/speculative_decoding/sharegpt_medusa/ShareGPT_V4.3_unfiltered_cleaned_split.json 

# vicuna 13b 8bit
python allocation.py --outdir train_data_vicuna13b \
                    --type vicuna \
                    --basemodel /root/MODELS/vicuna-13b-v1.3 \
                    --data /root/idea/speculative_decoding/sharegpt_medusa/ShareGPT_V4.3_unfiltered_cleaned_split.json 

# llama2 7b
python allocation.py --outdir train_data_llama27b \
                    --type vicuna \
                    --basemodel /root/MODELS/Llama-2-7b-chat-hf \
                    --data /root/idea/speculative_decoding/sharegpt_medusa/ShareGPT_V4.3_unfiltered_cleaned_split.json 

# llama2 13b 8bit
python allocation.py --outdir train_data_llama213b \
                    --type vicuna \
                    --basemodel /root/MODELS/Llama-2-13b-chat-hf \
                    --data /root/idea/speculative_decoding/sharegpt_medusa/ShareGPT_V4.3_unfiltered_cleaned_split.json 