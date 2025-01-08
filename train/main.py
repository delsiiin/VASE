# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# Adapted from: https://github.com/lm-sys/FastChat/blob/main/fastchat/train/train.py

from dataclasses import dataclass, field
import json
import math
import pathlib
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
import transformers
from transformers import Trainer, BitsAndBytesConfig
from transformers.trainer_pt_utils import LabelSmoother

from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
import os

from transformers import LlamaForCausalLM

from methods.ssd.model.ssd_model import SSDModel
from methods.ssd.model.ssd_config import SSDConfig

from typing import Any, Dict, List

import random

def seed_torch(seed=42):
 
    random.seed(seed)
 
    np.random.seed(seed)
 
    torch.manual_seed(seed)
 
    torch.cuda.manual_seed(seed)
 
    torch.cuda.manual_seed_all(seed) 
 
    torch.backends.cudnn.benchmark = False
 
    torch.backends.cudnn.deterministic = True

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

seed_torch(42)

# Customized for training Medusa heads

def replace_compute_loss_cross_entropy(
    ssd_decay_coefficient, 
):
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the training loss for the model.

        Args:
            model (torch.nn.Module): The model for which to compute the loss.
            inputs (dict): The input data, including input IDs, attention mask, and labels.
            return_outputs (bool): Whether to return model outputs along with the loss.

        Returns:
            Union[float, Tuple[float, torch.Tensor]]: The computed loss, optionally with model outputs.
        """
        # DDP will give us model.module
        if hasattr(model, "module"):
            topk = model.module.topk
        else:
            topk = model.topk

        logits = model(
            hidden_states=inputs["hidden_states"], attention_mask=inputs["attention_mask"]
        )
        labels = inputs["labels"]
        # Shift so that tokens < n predict n
        loss = 0
        loss_fct = CrossEntropyLoss()
        log = {}
        for i in range(topk):
            ssd_logits = logits[i, :, : -(2 + i)].contiguous()
            ssd_labels = labels[..., 2 + i :].contiguous()
            ssd_logits = ssd_logits.view(-1, logits.shape[-1])
            ssd_labels = ssd_labels.view(-1)
            ssd_labels = ssd_labels.to(ssd_logits.device)
            loss_i = loss_fct(ssd_logits, ssd_labels)
            loss += loss_i * ssd_decay_coefficient ** i * 0.2

            
            # not_ignore = ssd_labels.ne(IGNORE_TOKEN_ID)
            # ssd_labels = ssd_labels[not_ignore]

            # Add top-k accuracy
            # for k in range(1, 6):
            #     _, topk = ssd_logits.topk(k, dim=-1)
            #     topk = topk[not_ignore]
            #     correct = topk.eq(ssd_labels.unsqueeze(-1)).any(-1)
            #     log[f"ssd{i}_top{k}"] = correct.float().mean().item()

            # log[f"ssd{i}_loss"] = loss_i.item()

        log[f"loss"] = loss.item()
        self.log(log)
        return (loss, logits) if return_outputs else loss
    
    transformers.trainer.Trainer.compute_loss = compute_loss

def list_files(path):
    datapath = []
    for root, directories, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            datapath.append(file_path)
    return datapath

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="lmsys/vicuna-7b-v1.3")
   
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Load in 4 bit."},
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Load in 8 bit."},
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default="sharegpt_clean.json",
        metadata={"help": "Path to the training data."},
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = True
    model_type: str = field(
        default="vicuna",
        metadata={"help": "vicuna/llama2/llama3"},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    top_layers_len: int = field(
        default=24,
        metadata={"help": "Number of Medusa heads."},
    )
    top_k_group: int = field(
        default=4,
        metadata={"help": "Number of layers for each Medusa head."},
    )
    resnet_num: int = field(
        default=1,
        metadata={"help": "Number of layers for each Medusa head."},
    )
    attn_hid_dim: int = field(
        default=1024,
        metadata={"help": "Number of layers for each Medusa head."},
    )
    ssd_decay_coefficient: float = field(
        default=0.8,
        metadata={"help": "Number of layers for each Medusa head."},
    )
    remove_unused_columns: bool = field(default=False)


local_rank = None


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """
    Save the model's state dictionary to a specified directory.

    Args:
        trainer (transformers.Trainer): The Hugging Face Trainer object.
        output_dir (str): The directory where the model state dictionary will be saved.
    """
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


class CustomDataset(Dataset):
    def __init__(self, datapath):
        self.data = datapath

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # try:
        data = torch.load(self.data[index])
        hidden_state = data['hidden_state'][:2048][None, :]
        input_ids = data['input_ids'][:2048][None, :]
        labels = data["labels"][:2048][None, :]


        length = hidden_state.shape[1]
        # length_q = data['query_ids'].shape[1]
        attention_mask = [1] * length

        return dict(
            hidden_state_big=hidden_state,
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )


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


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False

    # Generate Medusa config for pushing to HF hub
    ssd_config = SSDConfig(
        top_layers_len=training_args.top_layers_len,
        top_k_group=training_args.top_k_group,
        resnet_num=training_args.resnet_num,
        attn_hid_dim=training_args.attn_hid_dim,
        **config.to_dict()  # Inherit all parameters from the base config
    )

    if data_args.model_type == "llama3":
        ssd_config.max_position_embeddings = training_args.model_max_length

    print(ssd_config)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    
    if data_args.model_type == "llama3":
        tokenizer.pad_token_id = 0
    else:
        tokenizer.pad_token = tokenizer.unk_token

    # Check if datasets already exist
    datapath = list_files(data_args.data_path)

    traindatapath = datapath[:int(len(datapath) * 0.99)]
    testdatapath = datapath[int(len(datapath) * 0.99):]

    traindataset = CustomDataset(traindatapath)
    testdataset = CustomDataset(testdatapath)

    data_module = dict(train_dataset=traindataset, eval_dataset=testdataset)

    # Format output dir
    training_args.output_dir = os.path.join(training_args.output_dir, f"{model_args.model_name_or_path.split('/')[-1]}_ssd_{training_args.top_k_group}_lr_{training_args.learning_rate}_dim_{training_args.attn_hid_dim}")

    # Save Medusa config
    ssd_config.save_pretrained(training_args.output_dir)

    # Add Medusa heads
    ssd_model = SSDModel(
        None,
        ssd_config,
        model_args.model_name_or_path,
    )

    replace_compute_loss_cross_entropy(training_args.ssd_decay_coefficient)

    trainer = Trainer(
        model=ssd_model, tokenizer=tokenizer, args=training_args, data_collator=DataCollatorWithPadding(), **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # model.config.use_cache = True
    # trainer.save_state()
    # safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    # Save MedusaHead seperately
    if hasattr(ssd_model, "module"):
        ssd_model = ssd_model.module.router
    else:
        ssd_model = ssd_model.router

    # Save Medusa heads
    torch.save(
        ssd_model.state_dict(),
        os.path.join(training_args.output_dir, "ssd_model.pt"),
    )


if __name__ == "__main__":
    train()