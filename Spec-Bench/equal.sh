# SSD
python evaluation/equal.py \
        --file-path /root/idea/speculative_decoding/VASE/Spec-Bench/data/spec_bench/model_answer \
        --jsonfile1 vicuna-7b-v1.3-vanilla-float16-temp-0.0.jsonl \
        --jsonfile2 vicuna-7b-v1.3-ssd-float16-temperature-0.0.jsonl

# Medusa
python evaluation/equal.py \
        --file-path /root/idea/speculative_decoding/VASE/Spec-Bench/data/spec_bench/model_answer \
        --jsonfile1 vicuna-7b-v1.3-vanilla-float16-temp-0.0.jsonl \
        --jsonfile2 vicuna-7b-v1.3-medusa-float16-temperature-0.0.jsonl