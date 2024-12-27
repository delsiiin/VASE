"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse

from evaluation.eval import run_eval, reorg_answer_file

from fastchat.utils import str_to_torch_dtype

from model.ssd.utils import *
from model.ssd.ssd_model import SSDModel
from model.ssd.kv_cache import initialize_past_key_values, initialize_past_key_values_attn
from model.ssd.medusa_choices import *

from transformers import AutoTokenizer

def ssd_forward(inputs, model, tokenizer, max_new_tokens, medusa_choices=None, temperature=0.0, posterior_threshold=0.09, posterior_alpha=0.3, max_steps=512):
    input_ids = inputs.input_ids
    assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
    # Avoid modifying the input_ids in-place
    input_ids = input_ids.clone()
    accept_length_list = []

    # Cache medusa buffers (the fixed patterns for tree attention)
    if hasattr(model, "medusa_choices") and model.medusa_choices == medusa_choices:
        # Load the cached medusa buffer
        ssd_buffers = model.ssd_buffers
    else:
        # Initialize the medusa buffer
        ssd_buffers = generate_ssd_buffers(
            medusa_choices, device=model.base_model.device
        )
    model.ssd_buffers = ssd_buffers
    model.medusa_choices = medusa_choices

    # Initialize the past key and value states
    if hasattr(model, "past_key_values"):
        past_key_values = model.past_key_values
        past_key_values_data = model.past_key_values_data
        current_length_data = model.current_length_data
        # Reset the past key and value states
        current_length_data.zero_()

        past_key_values_attn = model.past_key_values_attn
        past_key_values_data_attn = model.past_key_values_data_attn
        current_length_data_attn = model.current_length_data_attn
        # Reset the past key and value states
        current_length_data_attn.zero_()
    else:
        (
            past_key_values,
            past_key_values_data,
            current_length_data,
        ) = initialize_past_key_values(model)
        model.past_key_values = past_key_values
        model.past_key_values_data = past_key_values_data
        model.current_length_data = current_length_data

        past_key_values_attn, past_key_values_data_attn, current_length_data_attn = initialize_past_key_values_attn(model)

        model.past_key_values_attn = past_key_values_attn
        model.past_key_values_data_attn = past_key_values_data_attn
        model.current_length_data_attn = current_length_data_attn

    input_len = input_ids.shape[1]
    cur_length = input_len
    reset_ssd_mode(model, True)
    ssd_logits, logits = initialize_ssd(
            input_ids, model, ssd_buffers["ssd_attn_mask"], past_key_values, True, past_key_values_attn
    )
    new_token = 0
    
    for idx in range(max_steps): # idx: new decoding steps
        candidates, tree_candidates = generate_ssd_candidates(
                ssd_logits,
                logits,
                ssd_buffers["tree_indices"],
                ssd_buffers["retrieve_indices"],
            )
        ssd_logits, logits = ssd_tree_decoding(
                model,
                tree_candidates,
                past_key_values,
                ssd_buffers["ssd_position_ids"],
                input_ids,
                ssd_buffers["retrieve_indices"],
                True,
                past_key_values_attn
            )
        best_candidate, accept_length = evaluate_posterior(
                logits, candidates, temperature, posterior_threshold, posterior_alpha
            )
        input_ids, logits, ssd_logits, new_token = update_inference_inputs_ssd(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                ssd_buffers["retrieve_indices"],
                logits,
                ssd_logits,
                new_token,
                past_key_values_data,
                current_length_data,
                True,
                past_key_values_data_attn,
                current_length_data_attn
            )
        accept_length_tree = input_ids.shape[1] - cur_length
        cur_length = accept_length_tree + cur_length
        accept_length_list.append(accept_length_tree)
        if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            break
        if new_token > max_new_tokens:
            break
    return input_ids, new_token, idx+1, accept_length_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument("--base-model", type=str, default=None, help="Base model name or path.")
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=512,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The temperature for medusa sampling.",
    )
    parser.add_argument(
        "--posterior-threshold",
        type=float,
        default=0.09,
        help="The posterior threshold for medusa sampling.",
    )
    parser.add_argument(
        "--posterior-alpha",
        type=float,
        default=0.3,
        help="The posterior alpha for medusa sampling.",
    )
    parser.add_argument(
        "--medusa-choices",
        type=str,
        default="mc_sim_7b_63",
        help="The medusa choices for medusa sampling.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float64", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU.",
    )

    args = parser.parse_args()

    args.model_id = args.model_id+"-temperature-"+str(args.temperature)
    # args.medusa_choices = eval(args.medusa_choices)
    args.medusa_choices = ssd_vicuna_7b_v13_24_3

    question_file = f"data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    model = SSDModel.from_pretrained(
        args.base_model,
        args.model_path,
        torch_dtype=str_to_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        device_map="auto"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    run_eval(
        model=model,
        tokenizer=tokenizer,
        forward_func=ssd_forward,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_tokens=args.max_new_tokens,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        medusa_choices=args.medusa_choices,
        max_steps=args.max_steps,
        temperature=args.temperature,
        posterior_threshold=args.posterior_threshold,
        posterior_alpha=args.posterior_alpha,
    )

    reorg_answer_file(answer_file)