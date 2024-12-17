import argparse

parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--basepath', type=str, default='/home/zmw/vicuna-7b-v1.3')
parser.add_argument('--load_in_4bit', default=False, action='store_true')
parser.add_argument('--load_in_8bit', default=False, action='store_true')
parser.add_argument('--configpath', type=str, default="config.json")
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--bs', type=int, default=1)
parser.add_argument('--gradient-accumulation-steps', type=int, default=4)
parser.add_argument('--tmpdir', type=str, default='/home/zmw/sharegpt_medusa/ShareGPT_V4.3_unfiltered_cleaned_split.json')
parser.add_argument('--cpdir', type=str, default='./test')
parser.add_argument('--top_layers_len', type=int, default=24)
parser.add_argument('--top_k_group', type=int, default=4)
parser.add_argument('--resnet_num', type=int, default=1)
parser.add_argument('--attn_hid_dim', type=int, default=1024)
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
    "max_len": 1600,
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


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    Preprocesses conversation data and tokenizes it for model input.

    Args:
        sources: A list of conversation sources.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for tokenization.

    Returns:
        Dict: A dictionary containing tokenized inputs, labels, and attention mask.
    """
    conv = get_conversation_template("vicuna")
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}, {j}, {role}, {conv.roles[j % 2]}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO

    # Mask targets. Only compute loss on the assistant outputs.
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        turns = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, turn in enumerate(turns):
            if turn == "":
                break
            turn_len = len(tokenizer(turn).input_ids)

            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            # "-2" is hardcoded for the LLaMA tokenizer to make the offset correct.
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            # Ignore the user instructions
            target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
            cur_len += turn_len

        target[cur_len:] = IGNORE_TOKEN_ID

        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            rank0_print(tokenizer.decode(z))

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                rank0_print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning.

    Args:
        raw_data (list): A list of raw data examples.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for data preprocessing.
    """

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Lazy dataset for supervised fine-tuning.

    This dataset loads data on-the-fly when requested, which can be memory-efficient but slower.

    Args:
        raw_data (list): A list of raw data examples.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for data preprocessing.
    """

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.basepath,
        model_max_length=train_config['max_len'],
        padding_side="right",
        use_fast=True,
    )
tokenizer.pad_token = tokenizer.unk_token



# Check if datasets already exist
train_dataset_path = os.path.join(args.cpdir, "train_dataset.pt")
eval_dataset_path = os.path.join(args.cpdir, "eval_dataset.pt")

if os.path.exists(train_dataset_path) and os.path.exists(eval_dataset_path):
    # Load the datasets
    traindataset = torch.load(train_dataset_path)
    testdataset = torch.load(eval_dataset_path)

else:
    # Load data
    data = json.load(open(train_config['datapath'], "r"))

    traindata = data[:int(len(data) * 0.99)]
    testdata = data[int(len(data) * 0.99):]

    traindataset = SupervisedDataset(traindata, tokenizer)
    testdataset = SupervisedDataset(testdata, tokenizer)

    # Filter out elements where labels are all -100
    def filter_dataset(dataset):
        filtered_data = []
        for data in dataset:
            if not torch.all(data["labels"] == IGNORE_TOKEN_ID):
                filtered_data.append(data)
        return filtered_data

    traindataset = filter_dataset(traindataset)
    testdataset = filter_dataset(testdataset)

    # Save the datasets for future use
    torch.save(traindataset, train_dataset_path)
    torch.save(testdataset, eval_dataset_path)


train_loader = DataLoader(traindataset, batch_size=train_config["bs"], shuffle=True,
                            num_workers=train_config["num_workers"],
                          pin_memory=True)
test_loader = DataLoader(testdataset, batch_size=train_config["bs"], shuffle=False,
                            num_workers=train_config["num_workers"], pin_memory=True)


if accelerator.is_main_process:
    if not os.path.exists(args.cpdir):
        os.makedirs(args.cpdir)

# Set RoPE scaling factor
config = transformers.AutoConfig.from_pretrained(
    args.basepath,
)
config.use_cache = False

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# Load model and tokenizer
model = transformers.AutoModelForCausalLM.from_pretrained(
    args.basepath,
    config=config,
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config if args.load_in_4bit else None,
    load_in_4bit=args.load_in_4bit,
    load_in_8bit=args.load_in_8bit,
)

# Generate Medusa config for pushing to HF hub
ssd_config = SSDConfig(
    top_layers_len=args.top_layers_len,
    top_k_group=args.top_k_group,
    resnet_num=args.resnet_num,
    attn_hid_dim=args.attn_hid_dim
)

# Save Medusa config
ssd_config.save_pretrained(args.cpdir)

# Add Medusa heads
ssd_model = SSDModel(
    model,
    ssd_config
)

# Freeze the base model
for param in ssd_model.base_model.parameters():
    param.requires_grad = False

optimizer = optim.AdamW(ssd_model.parameters(), lr=train_config["lr"], betas=(train_config["b1"], train_config["b2"]))

num_epochs = train_config["num_epochs"]
num_warmup_steps = train_config["num_warmup_steps"]
total_steps = train_config["total_steps"]
is_warmup = train_config["is_warmup"]

if is_warmup:
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=total_steps)

    ssd_model, model, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
        ssd_model, model, optimizer, train_loader, test_loader, scheduler
    )
else:
    ssd_model, model, optimizer, train_loader, test_loader = accelerator.prepare(
        ssd_model, model, optimizer, train_loader, test_loader
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
            predict = ssd_model(input_ids=data["input_ids"], attention_mask=data["attention_mask"])#这里hidden states是在数据预处理时就用大模型计算好了的

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

                predict = ssd_model(input_ids=data["input_ids"],
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
            