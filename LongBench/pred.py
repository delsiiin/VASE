import os
from datasets import load_dataset, load_from_disk
import torch
import json
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import random
import argparse
from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
import torch.distributed as dist
import torch.multiprocessing as mp

from methods.ssd.model.medusa_choices import *
from methods.ssd.model.utils import *
from methods.medusa.model.utils import *
from methods.ssd.model.kv_cache import *
# from medusa.model.utils import *
from peft import PeftModel, PeftConfig

import time

from methods.medusa.model.medusa_model import MedusaModel
# from medusa.model.utils import *

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, choices=["llama2-7b-chat-4k", "longchat-v1.5-7b-32k", "xgen-7b-8k", "internlm-7b-8k", "chatglm2-6b", "chatglm2-6b-32k", "chatglm3-6b-32k", "vicuna-v1.5-7b-16k", "vicuna-v1.3-7b", "medusa-vicuna-7b-v1.3"], help="Model name")
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument("--dataset", type=str, required=False, default="qasper")

    parser.add_argument("--ssd_name", type=str, required=False, help="Model name or path.", default='/root/idea/speculative_decoding/ssd_hand/train/ssd_models/vicuna-7b-v1.3_ssd_3_lr_0.002_dim_1024')
    
    parser.add_argument("--attn", action='store_true', required=False, default=False)

    parser.add_argument("--medusa", action='store_true', required=False, default=False)

    parser.add_argument("--tree", action='store_true', required=False, default=False)
    
    return parser.parse_args(args)

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def medusa_generate(model, model_name, tokenizer, input_ids, max_new_tokens, device):

    if "vicuna" in model_name:
        if "v1.3" in model_name:
            max_seq_length = 2048
        elif "v1.5" in model_name:
            max_seq_length = 4096
        past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(model)
    elif "llama" in model_name:
        max_seq_length = 4096
        past_key_values, past_key_values_data, current_length_data = initialize_past_key_values_llama(model)
    
    model.past_key_values = past_key_values
    model.past_key_values_data = past_key_values_data
    model.current_length_data = current_length_data

    model.current_length_data.zero_() # this is for rerun

    input_len = len(input_ids[0])
    # print('Input token length:', len(input_ids[0]))
    # print('Init KV cache shape for attention modules:', model.past_key_values[0][0].shape, model.past_key_values[0][1].shape)

    output_token = torch.tensor([], dtype=torch.long).to(device)

    inference_count = 0
    accept_lengths = []
    with torch.inference_mode():

        medusa_logits, outputs, logits = model(input_ids, output_orig = True, past_key_values=model.past_key_values, medusa_forward=True)
        inference_count += 1

        medusa_pred = torch.argmax(medusa_logits[..., -1, :], dim = -1)
        pred = torch.argmax(logits[..., -1, :], dim = -1)
        preds = torch.cat([pred, medusa_pred[:, 0 ]], dim = -1)
        # print(f'Prediction @ {inference_count}: {tokenizer.batch_decode(pred)}')
        output_token = torch.cat([output_token, pred], dim = -1)

        cur_length = input_len
        accept_lengths.append(1)
        step = 0
        for _ in range(max_new_tokens):
            
            if step >= max_new_tokens:
                break

            medusa_logits, outputs, logits = model(preds.cuda().unsqueeze(0), output_orig = True, past_key_values = model.past_key_values, medusa_forward=True)
            inference_count += 1

            medusa_pred = torch.argmax(medusa_logits[..., -6:, :], dim = -1)
            pred = torch.argmax(logits[..., :, :], dim = -1)
            posterior_mask = (
                        preds[1:] == pred[0, :-1]
                    ).int()
            accept_length = torch.cumprod(posterior_mask, dim = -1).sum().item()
            cur_length = cur_length + accept_length + 1
            # update kv cache
            model.current_length_data.fill_(cur_length)
            # create new input
            preds = torch.cat([pred[:, accept_length], medusa_pred[:,0,accept_length]], dim = -1)
            output_token = torch.cat([output_token, pred[0, :accept_length + 1]], dim = -1)
            # print(f'Prediction @ {inference_count}: {tokenizer.batch_decode(pred[0, :accept_length + 1])}')
            accept_lengths.append(accept_length + 1)
            step += accept_length + 1
            if tokenizer.eos_token_id in pred[0, :accept_length + 1] or cur_length + medusa_pred.shape[0] >= max_seq_length:
                break
    
    print(f'Final output: {tokenizer.decode(output_token, skip_special_tokens=True)}')

    # plt.plot(accept_lengths)
    # plt.xlabel('Inference step')
    # plt.ylabel('Accept length')
    # plt.savefig('accept_length.png')
    print('Avg. accept length:', np.mean(accept_lengths))
    print('Token num:', step)

    return tokenizer.decode(output_token, skip_special_tokens=True)

def medusa_tree_generate(model, model_name, tokenizer, input_ids, max_new_tokens, device):

    if "vicuna" in model_name:
        if "v1.3" in model_name:
            max_seq_length = 2048
        elif "v1.5" in model_name:
            max_seq_length = 4096
        past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(model)
    elif "llama" in model_name:
        max_seq_length = 4096
        past_key_values, past_key_values_data, current_length_data = initialize_past_key_values_llama(model)
    
    model.past_key_values = past_key_values
    model.past_key_values_data = past_key_values_data
    model.current_length_data = current_length_data

    model.current_length_data.zero_() # this is for rerun

    input_len = len(input_ids[0])
    # print('Input token length:', len(input_ids[0]))
    # print('Init KV cache shape for attention modules:', model.past_key_values[0][0].shape, model.past_key_values[0][1].shape)
    medusa_choices = mc_sim_7b_63

    accept_lengths_tree = []
    with torch.inference_mode():

        new_token = 0

        reset_medusa_mode(model)
        medusa_buffers = generate_medusa_buffers(
                    medusa_choices, device=model.base_model.device
                )

        medusa_logits, logits = initialize_medusa(
            input_ids, model, medusa_buffers["medusa_attn_mask"], past_key_values
        )

        cur_length = input_len + 1
        accept_lengths_tree.append(1)
        step = 0
        for _ in range(max_new_tokens):

            if step >= max_new_tokens:
                break

            candidates, tree_candidates = generate_candidates(
                    medusa_logits,
                    logits,
                    medusa_buffers["tree_indices"],
                    medusa_buffers["retrieve_indices"],
                )
            medusa_logits, logits, outputs = tree_decoding(
                    model,
                    tree_candidates,
                    past_key_values,
                    medusa_buffers["medusa_position_ids"],
                    input_ids,
                    medusa_buffers["retrieve_indices"],
                )
            best_candidate, accept_length = evaluate_posterior(
                    logits, candidates, temperature = 0, posterior_threshold = 0, posterior_alpha = 0
                )
            input_ids, logits, medusa_logits, new_token = update_inference_inputs(
                    input_ids,
                    candidates,
                    best_candidate,
                    accept_length,
                    medusa_buffers["retrieve_indices"],
                    outputs,
                    logits,
                    medusa_logits,
                    new_token,
                    past_key_values_data,
                    current_length_data,
                )
            
            accept_length_tree = input_ids.shape[1] - cur_length
            cur_length = accept_length_tree + cur_length
            accept_lengths_tree.append(accept_length_tree)
            step += accept_length_tree
            if model.tokenizer.eos_token_id in input_ids[0, input_len:] or cur_length + new_token >= max_seq_length:
                break
    
    print('Decode:', tokenizer.batch_decode(input_ids[:,input_len:])) 

    return tokenizer.batch_decode(input_ids[:,input_len:])[0]

def ssd_generate(model, model_name, tokenizer, input_ids, max_new_tokens, device, attn):

    if "vicuna" in model_name:
        if "v1.3" in model_name:
            max_seq_length = 2048
        elif "v1.5" in model_name:
            max_seq_length = 4096
        past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(model)

        model.past_key_values = past_key_values
        model.past_key_values_data = past_key_values_data
        model.current_length_data = current_length_data

        model.current_length_data.zero_() # this is for rerun

        if attn:
            past_key_values_attn, past_key_values_data_attn, current_length_data_attn = initialize_past_key_values_attn(model)

            model.past_key_values_attn = past_key_values_attn
            model.past_key_values_data_attn = past_key_values_data_attn
            model.current_length_data_attn = current_length_data_attn

            model.current_length_data_attn.zero_() # this is for rerun

    elif "llama" in model_name:
        max_seq_length = 4096
        past_key_values, past_key_values_data, current_length_data = initialize_past_key_values_llama(model)

        model.past_key_values = past_key_values
        model.past_key_values_data = past_key_values_data
        model.current_length_data = current_length_data

        model.current_length_data.zero_() # this is for rerun
    
    
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

            # print(draft_logits.shape)
            
            draft_pred = torch.argmax(draft_logits[..., (-model.config.top_k_group-1):, :], dim = -1)

            # print(draft_pred.shape)
            
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
                # print(model.current_length_data_attn)

            # create new input
            preds = torch.cat([pred[:, accept_length], draft_pred[:,0,accept_length]], dim = -1)
            output_token = torch.cat([output_token, pred[0, :accept_length + 1]], dim = -1)
            # preds = torch.cat([pred[:, accept_length], draft_pred[:accept_length,0,0]], dim = -1)
            # print(f'Prediction @ {inference_count}: {tokenizer.batch_decode(pred[0, :accept_length + 1])}')
            accept_lengths.append(accept_length + 1)
            step += accept_length + 1
            if tokenizer.eos_token_id in pred[0, :accept_length + 1] or cur_length + draft_pred.shape[0] >= max_seq_length:
                break
        
    print(f'Final output: {tokenizer.decode(output_token, skip_special_tokens=True)}')

    # plt.plot(accept_lengths)
    # plt.xlabel('Inference step')
    # plt.ylabel('Accept length')
    # plt.savefig('accept_length.png')
    print('Avg. accept length:', np.mean(accept_lengths))
    print('Token num:', step)

    return tokenizer.decode(output_token, skip_special_tokens=True)

def ssd_tree_generate(model, model_name, tokenizer, input_ids, max_new_tokens, attn, device):

    if "vicuna" in model_name:
        if "v1.3" in model_name:
            max_seq_length = 2048
        elif "v1.5" in model_name:
            max_seq_length = 4096
        past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(model)

        if attn:
            past_key_values_attn, past_key_values_data_attn, current_length_data_attn = initialize_past_key_values_attn(model)

            model.past_key_values_attn = past_key_values_attn
            model.past_key_values_data_attn = past_key_values_data_attn
            model.current_length_data_attn = current_length_data_attn

            model.current_length_data_attn.zero_() # this is for rerun
    elif "llama" in model_name:
        max_seq_length = 4096
        past_key_values, past_key_values_data, current_length_data = initialize_past_key_values_llama(model)

    model.past_key_values = past_key_values
    model.past_key_values_data = past_key_values_data
    model.current_length_data = current_length_data

    model.current_length_data.zero_() # this is for rerun
    
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
                logits, candidates, temperature = 0, posterior_threshold = 0, posterior_alpha = 0
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

    print('Decode:', tokenizer.batch_decode(input_ids[:,input_len:]))

    return tokenizer.batch_decode(input_ids[:,input_len:])[0]


def get_pred(data, max_length, max_gen, prompt_format, dataset, device, model_name, model2path, out_path, attn, ssd_name, medusa, tree):
    device = torch.device(f'cuda:{0}')
    model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device, attn, ssd_name, medusa, tree)
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)
        if "chatglm3" in model_name:
            if dataset in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            else:
                input = prompt.to(device)
        else:
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]
        # print(input.input_ids)
        if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            
            if attn:
                if tree:
                    pred = ssd_tree_generate(model, model_name, tokenizer, input.input_ids, max_gen, attn, device)
                else:
                    pred = ssd_generate(model, model_name, tokenizer, input.input_ids, max_gen, device, attn)

            elif medusa:
                if tree:
                    pred = medusa_tree_generate(model, model_name, tokenizer, input.input_ids, max_gen, device)
                else:
                    pred = medusa_generate(model, model_name, tokenizer, input.input_ids, max_gen, device)

            else:
                output = model.generate(
                    **input,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    min_length=context_length+1,
                    eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
                )[0]

                pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)

        else:
           
            if attn:
                if tree:
                    pred = ssd_tree_generate(model, model_name, tokenizer, input.input_ids, max_gen, attn, device)
                else:
                    pred = ssd_generate(model, model_name, tokenizer, input.input_ids, max_gen, device, attn)

            elif medusa:
                if tree:
                    pred = medusa_tree_generate(model, model_name, tokenizer, input.input_ids, max_gen, device)
                else:
                    pred = medusa_generate(model, model_name, tokenizer, input.input_ids, max_gen, device)

            else:
                output = model.generate(
                    **input,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                )[0]

                pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)

        pred = post_process(pred, model_name)
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')
    # dist.destroy_process_group()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path, model_name, device, attn, ssd_name, medusa, tree):
    if "chatglm" in model_name or "internlm" in model_name or "xgen" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    elif "llama2" in model_name:
        # replace_llama_attn_with_flash_attn()

        if attn:
        
            from methods.ssd.model.ssd_model import SSDModel

            model = SSDModel.from_pretrained(
                path,
                ssd_name,
                torch_dtype=torch.float16,
                device=device,
                )
            model = model.to(device)

            tokenizer = AutoTokenizer.from_pretrained(path)

        else:

            from transformers import LlamaForCausalLM
            # replace_llama_attn_with_flash_attn()
            model = LlamaForCausalLM.from_pretrained(
                path,
                torch_dtype=torch.float16,
                )
            model = model.to(device)

            tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
            
    elif "longchat" in model_name or "vicuna" in model_name:
        if attn:
        
            from methods.ssd.model.ssd_model import SSDModel

            model = SSDModel.from_pretrained(
                path,
                ssd_name,
                torch_dtype=torch.float16,
                device=device,
                )
            model = model.to(device)

            tokenizer = AutoTokenizer.from_pretrained(path)

        elif medusa:

            model = MedusaModel.from_pretrained(
                path,
                torch_dtype=torch.float16,
            )
            model = model.to(device)
            tokenizer = model.get_tokenizer()

        else:

            from transformers import LlamaForCausalLM
            # replace_llama_attn_with_flash_attn()
            model = LlamaForCausalLM.from_pretrained(
                path,
                torch_dtype=torch.float16,
                )
            model = model.to(device)

            tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
            

    model = model.eval()
    return model, tokenizer

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    # world_size = torch.cuda.device_count()
    # mp.set_start_method('spawn', force=True)

    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model
    # define your model
    max_length = model2maxlen[model_name]
    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
                    "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
                    "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")
    for dataset in [args.dataset]:
        if args.e:
            data = load_dataset('/root/idea/speculative_decoding/ssd_hand/evaluation/LongBench/LongBench', f"{dataset}_e", split='test')
            if not os.path.exists(f"pred_e/{model_name}"):
                os.makedirs(f"pred_e/{model_name}")
            out_path = f"pred_e/{model_name}/{dataset}.jsonl"
        else:
            data = load_dataset('/root/idea/speculative_decoding/ssd_hand/evaluation/LongBench/LongBench', dataset, split='test')
            if not os.path.exists(f"pred/{model_name}"):
                os.makedirs(f"pred/{model_name}")
            out_path = f"pred/{model_name}/{dataset}.jsonl"
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        # data_subsets = [data_all[i::world_size] for i in range(world_size)]
    
        get_pred(data_all, max_length, \
                        max_gen, prompt_format, dataset, device, model_name, model2path, out_path, args.attn, args.ssd_name, args.medusa, args.tree)