Vicuna_PATH=/root/MODELS/Llama-2-7b-chat-hf
Eagle_PATH=/your_own_path/EAGLE-Vicuna-7B-v1.3
Medusa_PATH=/root/MODELS/medusa-vicuna-7b-v1.3
Hydra_PATH=/your_own_path/hydra-vicuna-7b-v1.3
Drafter_PATH=/your_own_path/vicuna-68m
Space_PATH=/your_own_path/vicuna-v1.3-7b-space
SSD_PATH=/root/idea/speculative_decoding/VASE/train/ssd_models/Llama-2-7b-chat-hf_ssd_3_lr_0.002_dim_1024
datastore_PATH=./model/rest/datastore/datastore_chat_large.idx
MODEL_NAME=llama2-7b-chat
TEMP=0.0
GPU_DEVICES=0

bench_NAME="spec_bench"
torch_dtype="float16" # ["float32", "float64", "float16", "bfloat16"]

# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_baseline --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-vanilla-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_sps --model-path $Vicuna_PATH --drafter-path $Drafter_PATH --model-id ${MODEL_NAME}-sps-68m-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_medusa --model-path $Medusa_PATH --base-model $Vicuna_PATH --model-id ${MODEL_NAME}-medusa-${torch_dtype} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_eagle --ea-model-path $Eagle_PATH --base-model-path $Vicuna_PATH --model-id ${MODEL_NAME}-eagle-${torch_dtype} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_eagle2 --ea-model-path $Eagle_PATH --base-model-path $Vicuna_PATH --model-id ${MODEL_NAME}-eagle2-${torch_dtype} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} USE_LADE=1 python -m evaluation.inference_lookahead --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-lade-level-5-win-7-guess-7-${torch_dtype} --level 5 --window 7 --guess 7 --bench-name $bench_NAME --dtype $torch_dtype
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_pld --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-pld-${torch_dtype} --bench-name $bench_NAME --dtype $torch_dtype
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} RAYON_NUM_THREADS=6 python -m evaluation.inference_rest --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-rest-${torch_dtype} --datastore-path $datastore_PATH --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_hydra --model-path $Hydra_PATH --base-model $Vicuna_PATH --model-id ${MODEL_NAME}-hydra-${torch_dtype} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_space --model-path $Space_PATH --model-id ${MODEL_NAME}-space-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype

CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_ssd --model-path $SSD_PATH --base-model $Vicuna_PATH --model-id ${MODEL_NAME}-ssd-${torch_dtype} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype 
