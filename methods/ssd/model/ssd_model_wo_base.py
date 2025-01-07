import copy
import json
import time

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig,AutoConfig

from transformers import Trainer, BitsAndBytesConfig


from transformers import AutoTokenizer
import os
from huggingface_hub import hf_hub_download

from huggingface_hub import hf_hub_download

from .ssd_config import SSDConfig

# from transformers import LlamaForCausalLM

from .modeling_llama_ssd import LlamaForCausalLM as SSDLlamaForCausalLM

from .router import RouterModel

class SSDModel(nn.Module):

    def __init__(
            self,
            base_model=None,
            ssd_config=None,
            base_path=None,
    ):

        super().__init__()
        self.base_model = base_model

        self.config = ssd_config

        self.topk = self.config.top_k_group
        
        self.router = RouterModel(self.config, self.config.attn_hid_dim)

        if base_model is not None:
            self.device = base_model.model.layers[-1].self_attn.q_proj.weight.device
       
            self.router.to(self.base_model.dtype).to(self.device)

            self.dtype = self.base_model.dtype
        else:
            self.head = torch.nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

            try:
                from safetensors import safe_open
                with open(os.path.join(base_path, "model.safetensors.index.json"), "r") as f:
                    index_json = json.loads(f.read())
                    head_path = index_json["weight_map"]["lm_head.weight"]
                with safe_open(os.path.join(base_path, head_path),
                            framework="pt",
                            device="cpu") as f:
                    tensor_slice = f.get_slice("lm_head.weight")
                    vocab_size, hidden_dim = tensor_slice.get_shape()
                    tensor = tensor_slice[:, :hidden_dim].float()
            except:
                with open(os.path.join(base_path, "pytorch_model.bin.index.json"), "r") as f:
                    index_json = json.loads(f.read())
                    head_path = index_json["weight_map"]["lm_head.weight"]
                weights = torch.load(os.path.join(base_path, head_path))
                tensor = weights["lm_head.weight"].float()

            self.head.weight.data = tensor
            self.head.eval()

            for param in self.head.parameters():
                param.requires_grad = False

    @classmethod
    def from_pretrained(
            cls,
            base_model_path=None,
            ssd_model_path=None,
            Type="LLaMA",
            load_in_4bit=False,
            load_in_8bit=False,
            **kwargs,
    ):
        
        ssd_config = SSDConfig.from_pretrained(ssd_model_path)
            
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        
        base_model = SSDLlamaForCausalLM.from_pretrained(
            base_model_path, 
            quantization_config=quantization_config if load_in_4bit else None,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            **kwargs
        )

        model = cls(
            base_model,
            ssd_config
        )
        ssd_path = os.path.join(ssd_model_path, "ssd_model.pt")
        if os.path.exists(ssd_path):
            filename = ssd_path
        else:
            filename = hf_hub_download(ssd_model_path, "ssd_model.pt")
        ssd_state_dict = torch.load(filename, map_location=base_model.device)
        model.router.load_state_dict(ssd_state_dict, strict=False)

        return model

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,
            past_key_values=None,
            past_key_values_attn=None,
            output_orig=False,
            position_ids=None,
            use_cache=None,
            hidden_states=None,
    ):
        if self.base_model is not None:
            with torch.inference_mode():
                # Pass input through the base model
                outputs = self.base_model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    use_cache=use_cache
                )
                if output_orig:
                    orig = self.base_model.lm_head(outputs[0])
                
            hidden_states = outputs[0].clone()

        all_draft_hidden_states = self.router(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values_attn,
            position_ids=position_ids,
            use_cache=use_cache
        )[0]

        if self.base_model is not None:
            all_draft_logits = self.base_model.lm_head(all_draft_hidden_states)
        else:
            all_draft_logits = self.head(all_draft_hidden_states)

        if output_orig:
            return all_draft_logits, outputs, orig
        return all_draft_logits