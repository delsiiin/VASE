# vicuna 7b
# SSD
python evaluation/speed.py \
        --file-path /root/idea/speculative_decoding/VASE/Spec-Bench/data/spec_bench/model_answer/vicuna-7b-v1.3-ssd-float16-temperature-0.0.jsonl \
        --base-path /root/idea/speculative_decoding/VASE/Spec-Bench/data/spec_bench/model_answer/vicuna-7b-v1.3-vanilla-float16-temp-0.0.jsonl \
        --tokenizer-path /root/MODELS/vicuna-7b-v1.3 

# vicuna 7b
# Medusa
python evaluation/speed.py \
        --file-path /root/idea/speculative_decoding/VASE/Spec-Bench/data/spec_bench/model_answer/vicuna-7b-v1.3-medusa-float16-temperature-0.0.jsonl \
        --base-path /root/idea/speculative_decoding/VASE/Spec-Bench/data/spec_bench/model_answer/vicuna-7b-v1.3-vanilla-float16-temp-0.0.jsonl \
        --tokenizer-path /root/MODELS/vicuna-7b-v1.3 

# llama2 7b
# SSD
python evaluation/speed.py \
        --file-path /root/idea/speculative_decoding/VASE/Spec-Bench/data/spec_bench/model_answer/llama2-7b-chat-ssd-float16-temperature-0.0.jsonl \
        --base-path /root/idea/speculative_decoding/VASE/Spec-Bench/data/spec_bench/model_answer/llama2-7b-chat-vanilla-float16-temp-0.0.jsonl \
        --tokenizer-path /root/MODELS/Llama-2-7b-chat-hf