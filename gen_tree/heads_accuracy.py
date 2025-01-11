import os
import torch
import json
from contextlib import contextmanager
import numpy as np
# from medusa.model.medusa_model import MedusaModel

from methods.ssd.model.kv_cache import *
# from medusa.model.utils import *
from methods.ssd.model.medusa_choices import *
from copy import deepcopy
import matplotlib.pyplot as plt
import torch.nn.functional as F
from fastchat.model.model_adapter import get_conversation_template
from tqdm import tqdm
import argparse

from peft import PeftModel, PeftConfig

from methods.ssd.model.utils import *

from transformers import AutoTokenizer

def get_accuracies(ssd, logit):
    # get the correct counts of each head
    seq_len, choices, topk = ssd.shape
    results = []
    for choice in range(choices):
        results.append(ssd[:-choice - 1,choice].eq(logit[choice + 1:,0]))
    return results


def main(args):

    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    #     print(f"使用设备: {torch.cuda.get_device_name(0)}")
    # else:
    #     device = torch.device("cpu")
    #     print("CUDA 不可用，使用 CPU")

    if args.attn:
        
        from methods.ssd.model.ssd_model import SSDModel

        model = SSDModel.from_pretrained(
            args.model_path,
            args.ssd_name,
            torch_dtype=torch.float16,
            device_map="auto",
            )
        # model = model.to(device)

        tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    model = model.eval()

    data = json.load(open(args.data_path))

    if "vicuna" in args.model_name:
        past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(model)
        model.past_key_values = past_key_values
        model.past_key_values_data = past_key_values_data
        model.current_length_data = current_length_data

        if args.attn:
            past_key_values_attn, past_key_values_data_attn, current_length_data_attn = initialize_past_key_values_attn(model)

            model.past_key_values_attn = past_key_values_attn
            model.past_key_values_data_attn = past_key_values_data_attn
            model.current_length_data_attn = current_length_data_attn
    elif "llama" in args.model_name:
        past_key_values, past_key_values_data, current_length_data = initialize_past_key_values_llama(model)
        model.past_key_values = past_key_values
        model.past_key_values_data = past_key_values_data
        model.current_length_data = current_length_data

        if args.attn:
            past_key_values_attn, past_key_values_data_attn, current_length_data_attn = initialize_past_key_values_llama_attn(model)

            model.past_key_values_attn = past_key_values_attn
            model.past_key_values_data_attn = past_key_values_data_attn
            model.current_length_data_attn = current_length_data_attn\

    results = None

    for sample in tqdm((data)):
        if "vicuna" in args.model_name:
            conv = get_conversation_template("vicuna")
            conv.messages = []
            conv.append_message(conv.roles[0], sample["instruction"])
            conv.append_message(conv.roles[1], "")
            prompt = conv.get_prompt()
        elif "llama2" in args.model_name:
            conv = get_conversation_template("llama-2-chat")
            sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
            conv.system_message = sys_p
            conv.messages = []
            conv.append_message(conv.roles[0], sample["instruction"])
            conv.append_message(conv.roles[1], "")
            prompt = conv.get_prompt() + " "
        elif "llama3" in args.model_name:
            messages = [
                {"role": "system",
                 "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
            ]
            convroles=["user","assistant"]
            messages.append(
                {"role": convroles[0], "content": sample["instruction"]}
            )
            messages.append(
                {"role": convroles[1], "content": ""}
            )
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        steps = args.steps
        logits_ids = []
        ssd_topk_ids = []

        with torch.inference_mode():
            input_ids = tokenizer([prompt]).input_ids
            input_ids = torch.as_tensor(input_ids).cuda()
            model.current_length_data.zero_() # this is for rerun
            if args.attn:
                model.current_length_data_attn.zero_() # this is for rerun

            reset_ssd_mode(model, args.attn)
           
            if args.attn:
                draft_logits, _, logits = model(input_ids, output_orig = True, past_key_values=model.past_key_values, past_key_values_attn=model.past_key_values_attn)
            
            _, ssd_topk = draft_logits[...,-1,:].topk(20, dim=-1)

            input_id = logits[:, -1:].argmax(dim=-1)
            logits_ids.append(input_id.detach().cpu())
            ssd_topk_ids.append(ssd_topk.detach().cpu())
            for _ in range(steps):
                    
                if args.attn:
                    draft_logits, _, logits = model(input_id, output_orig = True, past_key_values=model.past_key_values, past_key_values_attn=model.past_key_values_attn)
                
                _, ssd_topk = draft_logits[...,-1,:].topk(20, dim=-1)
                input_id = logits[:, -1:].argmax(dim=-1)
                logits_ids.append(input_id.detach().cpu())
                ssd_topk_ids.append(ssd_topk.detach().cpu())
            logits_ids = torch.stack(logits_ids, dim=0)
            ssd_topk_ids = torch.stack(ssd_topk_ids, dim=0).squeeze(2)
            if results is None:
                results = get_accuracies(ssd_topk_ids, logits_ids)
            else:
                # cat sub results
                cur_results = get_accuracies(ssd_topk_ids, logits_ids)
                for i in range(len(results)):
                    results[i] = torch.cat((results[i], cur_results[i]), dim=0)

    save_path = os.path.join(args.save_dir, args.model_name + "-" + str(model.config.top_layers_len) + "-" + str(model.config.top_k_group) + "_heads_accuracy.pt")
    torch.save(results, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SSD Model Evaluator")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the pre-trained Medusa model.")
    parser.add_argument("--ssd_name", type=str, required=False, help="Model name or path.", default='/root/idea/speculative_decoding/ssd_hand/train/ssd_models/vicuna-7b-v1.3_ssd_3_lr_0.002_dim_1024')
    parser.add_argument("--model_name", type=str, required=True,
                        help="Name of the model.")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the evaluation data in JSON format.")
    parser.add_argument("--save_dir", type=str, default="./data",
                        help="Directory to save the results.")
    parser.add_argument("--steps", type=int, default=20,
                        help="Number of steps to run the model.")
    parser.add_argument("--attn", action='store_true', required=False, default=False)

    args = parser.parse_args()

    # If the save directory doesn't exist, create it
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    main(args)