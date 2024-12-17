"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import json
import os

import time
import shortuuid
from tqdm import tqdm

from fastchat.llm_judge.common import load_questions
from fastchat.model import get_conversation_template

import torch

#try:

from peft import PeftModel, PeftConfig
from methods.ssd.model.kv_cache import initialize_past_key_values, initialize_past_key_values_llama, initialize_past_key_values_attn, initialize_past_key_values_llama_attn
from methods.ssd.model.medusa_choices import *
from methods.ssd.model.utils import *
from transformers import AutoTokenizer
import numpy as np

from methods.medusa.model.medusa_choices import *
from methods.medusa.model.utils import *


def ssd_generate(model, model_name, tokenizer, input_ids, max_new_tokens, max_seq_length, attn, device):

    if "Llama-2" in model_name:
        past_key_values, past_key_values_data, current_length_data = initialize_past_key_values_llama(model)

    model.past_key_values = past_key_values
    model.past_key_values_data = past_key_values_data
    model.current_length_data = current_length_data

    model.current_length_data.zero_() # this is for rerun

    if attn:
        if "Llama-2" in model_name:
            past_key_values_attn, past_key_values_data_attn, current_length_data_attn = initialize_past_key_values_llama_attn(model)

        model.past_key_values_attn = past_key_values_attn
        model.past_key_values_data_attn = past_key_values_data_attn
        model.current_length_data_attn = current_length_data_attn

        model.current_length_data_attn.zero_() # this is for rerun
    
    input_len = len(input_ids[0])
    # print('Input token length:', len(input_ids[0]))
    # print('Init KV cache shape for attention modules:', model.past_key_values[0][0].shape, model.past_key_values[0][1].shape)

    output_token = torch.tensor([], dtype=torch.long).to(device)

    inference_count = 0
    accept_lengths = []
    with torch.inference_mode():

        
        if attn:
            draft_logits, _, base_logits = model(input_ids, output_orig = True, past_key_values=model.past_key_values, past_key_values_attn=model.past_key_values_attn)
        
        inference_count += 1

        draft_pred = torch.argmax(draft_logits[..., -1, :], dim = -1)
        
        pred = torch.argmax(base_logits[..., -1, :], dim = -1)
        
        preds = torch.cat([pred, draft_pred[:, 0 ]], dim = -1)
        # print(preds.shape)

        output_token = torch.cat([output_token, pred], dim = -1)

        cur_length = input_len
        accept_lengths.append(1)
        step = 0
        for _ in range(max_new_tokens):

            if step >= max_new_tokens:
                break

            if attn:
                draft_logits, _, base_logits = model(preds.cuda().unsqueeze(0), output_orig = True, past_key_values = model.past_key_values, past_key_values_attn=model.past_key_values_attn)
           
            inference_count += 1

            
            draft_pred = torch.argmax(draft_logits[..., (-model.config.top_k_group-1):, :], dim = -1)
            
            pred = torch.argmax(base_logits[..., :, :], dim = -1)
            
            posterior_mask = (
                        preds[1:] == pred[0, :-1]
                    ).int()
            accept_length = torch.cumprod(posterior_mask, dim = -1).sum().item()
            
            # print(accept_length)

            cur_length = cur_length + accept_length + 1
            # update kv cache
            model.current_length_data.fill_(cur_length)
            if attn:
                model.current_length_data_attn.fill_(cur_length)
            # create new input
            preds = torch.cat([pred[:, accept_length], draft_pred[:,0,accept_length]], dim = -1)
            output_token = torch.cat([output_token, pred[0, :accept_length + 1]], dim = -1)
            # preds = torch.cat([pred[:, accept_length], draft_pred[:accept_length,0,0]], dim = -1)
            # print(f'Prediction @ {inference_count}: {tokenizer.batch_decode(pred[0, :accept_length + 1])}')
            accept_lengths.append(accept_length + 1)
            step += accept_length + 1
            if tokenizer.eos_token_id in pred[0, :accept_length + 1] or cur_length + draft_pred.shape[0] >= max_seq_length:
                break
        
        input_ids = torch.cat((input_ids, output_token.unsqueeze(0)), dim=-1)

    # print(f'Final output: {tokenizer.decode(output_token, skip_special_tokens=True)}')

    # # plt.plot(accept_lengths)
    # # plt.xlabel('Inference step')
    # # plt.ylabel('Accept length')
    # # plt.savefig('accept_length.png')
    # print('Avg. accept length:', np.mean(accept_lengths))
    # print('Token num:', step)
    return input_ids, step

def ssd_tree_generate(model, model_name, tokenizer, input_ids, max_new_tokens, max_seq_length, attn, device):

    if "Llama-2" in model_name:
        past_key_values, past_key_values_data, current_length_data = initialize_past_key_values_llama(model)

    model.past_key_values = past_key_values
    model.past_key_values_data = past_key_values_data
    model.current_length_data = current_length_data

    model.current_length_data.zero_() # this is for rerun

    if attn:
        if "Llama-2" in model_name:
            past_key_values_attn, past_key_values_data_attn, current_length_data_attn = initialize_past_key_values_llama_attn(model)

        model.past_key_values_attn = past_key_values_attn
        model.past_key_values_data_attn = past_key_values_data_attn
        model.current_length_data_attn = current_length_data_attn

        model.current_length_data_attn.zero_() # this is for rerun
    
    input_len = len(input_ids[0])
    # print('Input token length:', len(input_ids[0]))
    # print('Init KV cache shape for attention modules:', model.past_key_values[0][0].shape, model.past_key_values[0][1].shape)

    ssd_choices = ssd_vicuna_7b_v13_24_3

    accept_lengths_tree = []
    with torch.inference_mode():

        new_token = 0

        reset_ssd_mode(model, attn)
        ssd_buffers = generate_ssd_buffers(
            ssd_choices, device=model.device
        )

        ssd_logits, logits = initialize_ssd(
            input_ids, model, ssd_buffers["ssd_attn_mask"], past_key_values, attn, past_key_values_attn
        )

        cur_length = input_len + 1
        accept_lengths_tree.append(1)
        step = 0
        for _ in range(max_new_tokens):

            if step >= max_new_tokens:
                break
            
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
                attn,
                past_key_values_attn
            )

            best_candidate, accept_length = evaluate_posterior(
                logits, candidates, temperature = 0, posterior_threshold = 0, posterior_alpha = 0.
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
                attn,
                past_key_values_data_attn,
                current_length_data_attn
            )

            accept_length_tree = input_ids.shape[1] - cur_length
            cur_length = accept_length_tree + cur_length
            accept_lengths_tree.append(accept_length_tree)
            step += accept_length_tree
            if tokenizer.eos_token_id in input_ids[0, input_len:] or cur_length + new_token >= max_seq_length:
                break

    # print('Decode:', tokenizer.batch_decode(input_ids[:,input_len:]))

    return input_ids, new_token

def run_eval(
        model_name,
        question_file,
        question_begin,
        question_end,
        answer_file,
        max_new_token,
        max_seq_length,
        num_choices,
        temperature,
        ssd_name,
        attn,
        tree
):
    questions = load_questions(question_file, question_begin, question_end)
    # random shuffle the questions to balance the loading
    # random.shuffle(questions)
    shuffled_ids = [q["question_id"] for q in questions]
    # with open(f"data/{args.bench_name}/model_ids/{args.model_id}.shuffled_ids", "w") as fout:
    #     json.dump(shuffled_ids, fout)

    # Split the question file into `num_gpus` files
    # assert num_gpus_total % num_gpus_per_model == 0
    # use_ray = num_gpus_total // num_gpus_per_model > 1

    # if use_ray:
    #     get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
    #         get_model_answers
    #     ).remote
    # else:
        
    get_answers_func = get_model_answers

    chunk_size = len(questions)
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model_name,
                questions[i: i + chunk_size],
                answer_file,
                max_new_token,
                max_seq_length,
                num_choices,
                temperature,
                ssd_name,
                attn,
                tree
            )
        )

    # if use_ray:
    #     ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
        model_name,
        questions,
        answer_file,
        max_new_token,
        max_seq_length,
        num_choices,
        temperature,
        ssd_name,
        attn,
        tree
):
    #temperature = 0.0

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用设备: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA 不可用，使用 CPU")

    if attn:
        
        from methods.ssd.model.ssd_model import SSDModel

        model = SSDModel.from_pretrained(
            model_name,
            ssd_name,
            torch_dtype=torch.float16,
            device=device,
            )
        model = model.to(device)

        tokenizer = AutoTokenizer.from_pretrained(model_name)

    else:

        from transformers import LlamaForCausalLM
        # replace_llama_attn_with_flash_attn()
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            )
        model = model.to(device)

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)

    # if temperature > 1e-5:
    #     logits_processor = prepare_logits_processor(temperature=temperature)
    # else:
    #     logits_processor = None

    model.eval()
    print('Check model training state:', model.training)

    question = questions[0]

    # warmup
    for _ in range(3):
        torch.manual_seed(0)
        conv = get_conversation_template("llama-2-chat")
        sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
        conv.system_message = sys_p
        turns = []
        new_tokens = []
        wall_time = []
        for j in range(len(question["turns"])):
            qs = question["turns"][j]
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt() + " "
            input_ids = tokenizer([prompt]).input_ids

            # try:
            torch.cuda.synchronize()
            start_time = time.time()

            if attn:

                if tree:

                    output_ids, new_token = ssd_tree_generate(
                        model,
                        model_name,
                        tokenizer,
                        torch.as_tensor(input_ids).cuda(),
                        max_new_token,
                        max_seq_length,
                        attn,
                        device=device
                    )
                
                else:

                    output_ids, new_token = ssd_generate(
                        model,
                        model_name,
                        tokenizer,
                        torch.as_tensor(input_ids).cuda(),
                        max_new_token,
                        max_seq_length,
                        attn,
                        device=device
                    )
            
            else:

                output_ids = model.generate(
                    torch.as_tensor(input_ids).cuda(),
                    max_new_tokens=max_new_token,
                    num_beams=1,
                    do_sample=False,
                    temperature=0.0,
                )

                new_token = 0

            torch.cuda.synchronize()
            total_time = time.time() - start_time
            output_ids = output_ids[0][len(input_ids[0]):]
            # be consistent with the template's stop_token_ids
            if conv.stop_token_ids:
                stop_token_ids_index = [
                    i
                    for i, id in enumerate(output_ids)
                    if id in conv.stop_token_ids
                ]
                if len(stop_token_ids_index) > 0:
                    output_ids = output_ids[: stop_token_ids_index[0]]

            output = tokenizer.decode(
                output_ids,
                spaces_between_special_tokens=False,
            )
            conv.stop_str = "</s>"
            if conv.stop_str and output.find(conv.stop_str) > 0:
                output = output[: output.find(conv.stop_str)]
            for special_token in tokenizer.special_tokens_map.values():
                if isinstance(special_token, list):
                    for special_tok in special_token:
                        output = output.replace(special_tok, "")
                else:
                    output = output.replace(special_token, "")
            output = output.strip()

            if conv.name == "xgen" and output.startswith("Assistant:"):
                output = output.replace("Assistant:", "", 1).strip()


            turns.append(output)
            new_tokens.append(int(new_token))
            wall_time.append(total_time)
            conv.messages[-1][-1] = output
    print('Warmup done')

    # questions=questions[6:]
    for question in tqdm(questions):

        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            conv = get_conversation_template("llama-2-chat")
            sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
            conv.system_message = sys_p
            turns = []
            new_tokens = []
            wall_time = []
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt() + " "
                input_ids = tokenizer([prompt]).input_ids

                try:
                    torch.cuda.synchronize()
                    start_time = time.time()

                    if attn:

                        if tree:

                            output_ids, new_token = ssd_tree_generate(
                                model,
                                model_name,
                                tokenizer,
                                torch.as_tensor(input_ids).cuda(),
                                max_new_token,
                                max_seq_length,
                                attn,
                                device=device
                            )
                        
                        else:

                            output_ids, new_token = ssd_generate(
                                model,
                                model_name,
                                tokenizer,
                                torch.as_tensor(input_ids).cuda(),
                                max_new_token,
                                max_seq_length,
                                attn,
                                device=device
                            )

                    else:

                        output_ids = model.generate(
                            torch.as_tensor(input_ids).cuda(),
                            max_new_tokens=max_new_token,
                            num_beams=1,
                            do_sample=False,
                            temperature=0.0,
                        )

                        new_token = 0

                    torch.cuda.synchronize()
                    total_time = time.time() - start_time
                    output_ids = output_ids[0][len(input_ids[0]):]

                    if conv.stop_token_ids:
                        stop_token_ids_index = [
                            i
                            for i, id in enumerate(output_ids)
                            if id in conv.stop_token_ids
                        ]
                        if len(stop_token_ids_index) > 0:
                            output_ids = output_ids[: stop_token_ids_index[0]]

                    output = tokenizer.decode(
                        output_ids,
                        spaces_between_special_tokens=False,
                    )
                    if conv.stop_str and output.find(conv.stop_str) > 0:
                        output = output[: output.find(conv.stop_str)]
                    for special_token in tokenizer.special_tokens_map.values():
                        if isinstance(special_token, list):
                            for special_tok in special_token:
                                output = output.replace(special_tok, "")
                        else:
                            output = output.replace(special_token, "")
                    output = output.strip()

                    if conv.name == "xgen" and output.startswith("Assistant:"):
                        output = output.replace("Assistant:", "", 1).strip()
                except RuntimeError as e:
                    print("ERROR question ID: ", question["question_id"])
                    output = "ERROR"

                turns.append(output)
                new_tokens.append(int(new_token))
                wall_time.append(total_time)
                conv.messages[-1][-1] = output
            # torch.cuda.empty_cache()
            choices.append({"index": i, "turns": turns, "new_tokens": new_tokens, "wall_time": wall_time})

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model": model_name,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=False, help="Model name or path.", default='/home/zmw/Llama-2-7b-chat-hf')
    parser.add_argument("--model_id", type=str, required=False)
    parser.add_argument(
        "--bench_name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question_begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question_end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer_file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max_new_token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num_choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--tree_choices",
        type=str,
        default="mc_sim_7b_63",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=4096
    )

    # ssd
    parser.add_argument("--ssd_name", type=str, required=False, help="Model name or path.", default='/home/zmw/ssd_hand/train/ssd_models/vicuna-7b-v1.3_ssd_4_lr_0.002_dim_1024')
    parser.add_argument("--attn", action='store_true', required=False, default=False)

    parser.add_argument("--tree", action='store_true', required=False, default=False)

    args = parser.parse_args()

    # args.model = args.model + "-temperature-" + str(args.temperature)
    # args.tree_choices = eval(args.tree_choices)
    # if args.num_gpus_total // args.num_gpus_per_model > 1:
    #     import ray

    #     ray.init()

    question_file = f"/root/idea/speculative_decoding/ssd_hand/evaluation/general_bench/data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        if args.attn:
            answer_file = f"./result/{args.bench_name}/{args.model_id}_attn.jsonl"
        else: 
            answer_file = f"./result/{args.bench_name}/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    run_eval(
        args.model_name,
        question_file,
        args.question_begin,
        args.question_end,
        answer_file,
        args.max_new_token,
        args.max_seq_length,
        args.num_choices,
        args.temperature,
        args.ssd_name,
        args.attn,
        args.tree
    )

    reorg_answer_file(answer_file)