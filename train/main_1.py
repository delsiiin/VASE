import argparse

parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--basepath', type=str, default='/root/MODELS/vicuna-7b-v1.3')
parser.add_argument('--load_in_4bit', default=False, action='store_true')
parser.add_argument('--load_in_8bit', default=False, action='store_true')
parser.add_argument('--configpath', type=str, default="config.json")
parser.add_argument('--lr', type=float, default=2e-3)
parser.add_argument('--bs', type=int, default=1)
parser.add_argument('--gradient-accumulation-steps', type=int, default=4)
parser.add_argument('--tmpdir', type=str, default='/root/idea/speculative_decoding/VASE/gen_data/train_data_vicuna7b')
parser.add_argument('--cpdir', type=str, default='./test')
parser.add_argument('--top_layers_len', type=int, default=24)
parser.add_argument('--top_k_group', type=int, default=3)
parser.add_argument('--resnet_num', type=int, default=1)
parser.add_argument('--attn_hid_dim', type=int, default=4096)
parser.add_argument('--ssd_decay_coefficient', type=float, default=0.8)


parser.add_argument('--recover', type=int, default=-1)
args = parser.parse_args()

train_config = {
    "lr": args.lr,
    "bs": args.bs,
    "gradient_accumulation_steps": args.gradient_accumulation_steps,
    "datapath": f"{args.tmpdir}",
    "is_warmup": True,
    "num_epochs": 20,
    "num_warmup_steps": 2000,
    "total_steps": 800000,
    "num_workers": 1,
    "max_len": 2048,
    "b1": 0.9,
    "b2": 0.95,
    "grad_clip": 0.5,
    "save_freq": 5,
    "step_save_freq": 2000
}

import sys
sys.path.append('./')##TODO##
from torch.nn.utils.rnn import pad_sequence
import json
from safetensors import safe_open
# from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModelForSequenceClassification
import os

import torch

import random

torch.backends.cuda.matmul.allow_tf32 = True
from accelerate import Accelerator
from accelerate.utils import set_seed

set_seed(0)
accelerator = Accelerator(mixed_precision='bf16',
                          gradient_accumulation_steps=train_config["gradient_accumulation_steps"])\


from methods.ssd.model.ssd_model import SSDModel
from methods.ssd.model.ssd_config import SSDConfig

from typing import Any, Dict, List

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
# import accelerate
import numpy as np
from transformers import get_linear_schedule_with_warmup, AutoConfig
from torch.nn import functional as F

import transformers

from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template

from transformers import Trainer, BitsAndBytesConfig

##set offline wanb
# if accelerator.is_main_process:
#     import wandb
#     os.environ["WANDB_API_KEY"] = '8e7cc624974f2c73dac4b63b94cad221f1a801ac' # api key
#     os.environ["WANDB_MODE"] = "offline"   # offline wandb
#     wandb.init()

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def list_files(path):
    datapath = []
    for root, directories, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            datapath.append(file_path)
    return datapath



class CustomDataset(Dataset):
    def __init__(self, datapath):
        self.data = datapath

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # try:
        data = torch.load(self.data[index])
        new_data = {}
        hidden_state = data['hidden_state'][:train_config["max_len"]][None, :]
        input_ids = data['input_ids'][:train_config["max_len"]][None, :]
        labels = data["labels"][:train_config["max_len"]][None, :]


        length = hidden_state.shape[1]
        # length_q = data['query_ids'].shape[1]
        attention_mask = [1] * length

        new_data["attention_mask"] = attention_mask
        new_data["labels"] = labels
        new_data["hidden_state_big"] = hidden_state
        new_data["input_ids"] = input_ids

        return new_data


class DataCollatorWithPadding:

    def paddingtensor(self, intensors, N):
        B, n, S = intensors.shape
        # padding_tensor = torch.zeros(B, N - n, S,dtype=intensors.dtype)
        padding_tensor = torch.zeros(B, N - n, S)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors
    
    def paddingtensor2D_ignore(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.full((B, N - n), IGNORE_TOKEN_ID, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item['hidden_state_big'].shape[1] for item in features)
        batch_input_ids = torch.cat([self.paddingtensor2D(item['input_ids'], max_length) for item in features])
        batch_hidden_states = torch.cat([self.paddingtensor(item['hidden_state_big'], max_length) for item in features])
        batch_labels = torch.cat([self.paddingtensor2D_ignore(item['labels'], max_length) for item in features])
        batch_attention_mask = torch.tensor(
            [item['attention_mask'] + [0] * (max_length - len(item['attention_mask'])) for item in features])
        # batch_loss_mask = torch.ones_like(batch_loss_mask)
        # batch_attention_mask=torch.ones_like(batch_attention_mask)
        batch = {
            "input_ids": batch_input_ids,
            "hidden_states": batch_hidden_states,
            "attention_mask": batch_attention_mask,
            "labels": batch_labels,
        }
        return batch


tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.basepath,
        model_max_length=train_config['max_len'],
        padding_side="right",
        use_fast=True,
    )
tokenizer.pad_token = tokenizer.unk_token


# Check if datasets already exist
datapath = list_files(train_config['datapath'])

traindatapath = datapath[:int(len(datapath) * 0.99)]
testdatapath = datapath[int(len(datapath) * 0.99):]

traindataset = CustomDataset(traindatapath)
testdataset = CustomDataset(testdatapath)


train_loader = DataLoader(traindataset, batch_size=train_config["bs"], shuffle=True,
                            collate_fn=DataCollatorWithPadding(), num_workers=train_config["num_workers"],
                          pin_memory=True)
test_loader = DataLoader(testdataset, batch_size=train_config["bs"], shuffle=False,
                            collate_fn=DataCollatorWithPadding(), num_workers=train_config["num_workers"], pin_memory=True)


if accelerator.is_main_process:
    if not os.path.exists(args.cpdir):
        os.makedirs(args.cpdir)

# Set RoPE scaling factor
config = transformers.AutoConfig.from_pretrained(
    args.basepath,
)
config.use_cache = False

# Generate Medusa config for pushing to HF hub
ssd_config = SSDConfig(
    top_layers_len=args.top_layers_len,
    top_k_group=args.top_k_group,
    resnet_num=args.resnet_num,
    attn_hid_dim=args.attn_hid_dim,
    **config.to_dict()
)

# Save Medusa config
ssd_config.save_pretrained(args.cpdir)

# Add Medusa heads
ssd_model = SSDModel(
    None,
    ssd_config,
    args.basepath,
)

optimizer = optim.AdamW(ssd_model.parameters(), lr=train_config["lr"], betas=(train_config["b1"], train_config["b2"]))

num_epochs = train_config["num_epochs"]
num_warmup_steps = train_config["num_warmup_steps"]
total_steps = train_config["total_steps"]
is_warmup = train_config["is_warmup"]

if is_warmup:
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=total_steps)

    ssd_model, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
        ssd_model, optimizer, train_loader, test_loader, scheduler
    )
else:
    ssd_model, optimizer, train_loader, test_loader = accelerator.prepare(
        ssd_model, optimizer, train_loader, test_loader
    )


if args.recover != -1:
    accelerator.load_state("{}/state_{}".format(args.cpdir,args.recover))
    print("recover from checkpoint {}".format(args.recover))
    
for epoch in range(args.recover+1,num_epochs + 1):
    
    epoch_loss = 0
    num_batches = 0
    ssd_model.train()
    for batch_idx, data in enumerate(tqdm(train_loader)):

        with accelerator.accumulate(ssd_model):
            optimizer.zero_grad()
            predict = ssd_model(hidden_states=data["hidden_states"], attention_mask=data["attention_mask"])#这里hidden states是在数据预处理时就用大模型计算好了的

            labels = data["labels"]
            # Shift so that tokens < n predict n
            loss = 0
            loss_fct = nn.CrossEntropyLoss()
            log = {}
            for i in range(args.top_k_group):
                ssd_logits = predict[i, :, : -(2 + i)].contiguous()
                ssd_labels = labels[..., 2 + i :].contiguous()
                ssd_logits = ssd_logits.view(-1, predict.shape[-1])
                ssd_labels = ssd_labels.view(-1)
                ssd_labels = ssd_labels.to(ssd_logits.device)
                loss_i = loss_fct(ssd_logits, ssd_labels)
                loss += loss_i * args.ssd_decay_coefficient ** i * 0.2         
            
            
            # loss.backward()
            accelerator.backward(loss)
            accelerator.clip_grad_value_(ssd_model.parameters(), train_config["grad_clip"])
            optimizer.step()
            if is_warmup:
                scheduler.step()

        
        if accelerator.is_main_process:
            logdict = {"train/lr": optimizer.optimizer.param_groups[0]["lr"], "train/loss": loss.item()}
            print(logdict)
            #wandb.log(logdict)


        epoch_loss += loss.item()
        num_batches += 1

    epoch_loss /= num_batches
   
    # if accelerator.is_local_main_process:
    #     for id, i in enumerate(top_3acc):
    #         wandb.log({f'train/epochtop_{id + 1}_acc': i.sum().item() / total})
    if accelerator.is_local_main_process:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
        # print('Train Accuracy: {:.2f}%'.format(100 * correct / total))
        #wandb.log({"train/epochacc": correct / total, "train/epochloss": epoch_loss})

    #TEST
    if (epoch + 1) % train_config["save_freq"]:
       
        epoch_loss = 0
        num_batches = 0
        ssd_model.eval()

        for batch_idx, data in enumerate(tqdm(test_loader)):
            with torch.no_grad():

                predict = ssd_model(hidden_states=data["hidden_states"],
                                attention_mask=data["attention_mask"])
                labels = data["labels"]
                # Shift so that tokens < n predict n
                loss = 0
                loss_fct = nn.CrossEntropyLoss()
                log = {}
                for i in range(args.top_k_group):
                    ssd_logits = predict[i, :, : -(2 + i)].contiguous()
                    ssd_labels = labels[..., 2 + i :].contiguous()
                    ssd_logits = ssd_logits.view(-1, predict.shape[-1])
                    ssd_labels = ssd_labels.view(-1)
                    ssd_labels = ssd_labels.to(ssd_logits.device)
                    loss_i = loss_fct(ssd_logits, ssd_labels)
                    loss += loss_i * args.ssd_decay_coefficient ** i * 0.2    

            epoch_loss += loss.item()
            num_batches += 1

        # if accelerator.is_local_main_process:
        #     for id, i in enumerate(top_3acc):
        #         wandb.log({f'test/top_{id + 1}_acc': i.sum().item() / total})
        epoch_loss /= num_batches
        if accelerator.is_local_main_process:
            print('Test Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
            # print('Test Accuracy: {:.2f}%'.format(100 * correct / total))
            #wandb.log({"test/epochacc": correct / total, "test/epochloss": epoch_loss})
            accelerator.save_state(output_dir=f"{args.cpdir}/state_{epoch}")
            