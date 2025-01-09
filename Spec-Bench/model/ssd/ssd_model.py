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
    
    def tree_forward(
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
        if output_orig:
            with torch.inference_mode():
                # Pass input through the base model
                outputs = self.base_model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    use_cache=use_cache
                )
                orig = self.base_model.lm_head(outputs[0])
                
            hidden_states = outputs[0].clone()

        else:
            all_draft_hidden_states = self.router(
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values_attn,
                position_ids=position_ids,
                use_cache=use_cache
            )[0]

            all_draft_logits = self.base_model.lm_head(all_draft_hidden_states)

        if output_orig:
            return outputs, orig, hidden_states
        return all_draft_logits
    
    # eagle modified
    def topK_generate(
            self, 
            hidden_states=None,
            attention_mask=None,
            position_ids=None,
            past_key_values_attn=None,
            output_orig=False,
            orig=None,
            logits_processor=None
        ):
        
        scores_list, parents_list, ss_token = self.router.topK_forward(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values_attn,
            head=self.base_model.lm_head
        )

        scores_list = torch.cat(scores_list, dim=0).view(-1) # (64 * 2 + 8)

        ss_token_list = torch.cat(ss_token, dim=0).view(-1) # (64 * 2 + 8)
        
        top_scores = torch.topk(scores_list, self.router.total_tokens, dim=-1)
        top_scores_index = top_scores.indices # (63 total_tokens)
     
        top_scores_index = torch.sort(top_scores_index).values

        # print(ss_token_list)

        draft_tokens = ss_token_list[top_scores_index] # (63 total_tokens)

        base_token = torch.argmax(orig[:, -1]).unsqueeze(0)

        draft_tokens = torch.cat((base_token, draft_tokens), dim=0)

        draft_parents = torch.cat(parents_list, dim=0)[top_scores_index // self.router.top_k_draft].long() # (63 total_tokens)

        # print(draft_parents)
    
        mask_index = torch.searchsorted(top_scores_index, draft_parents - 1, right=False)
        # mask_index[(top_scores_index[mask_index]!=draft_parents - 1)]=-1
        mask_index[draft_parents == 0] = -1
        mask_index = mask_index + 1
        mask_index_list = mask_index.tolist()
       
        # with Timer("mask"):
        tree_mask = torch.eye(self.router.total_tokens + 1).bool()
        tree_mask[:, 0] = True
        for i in range(self.router.total_tokens):
            tree_mask[i + 1].add_(tree_mask[mask_index_list[i]])

        
        tree_position_ids = torch.sum(tree_mask, dim=1) - 1

        tree_mask = tree_mask.float()[None, None]
        
        draft_tokens = draft_tokens[None]
        # draft_tokens_ = tokenizer.decode(draft_tokens[0], skip_special_tokens=True)
        # print(draft_tokens_)

        del parents_list, scores_list, ss_token, ss_token_list, draft_parents

        max_depth = torch.max(tree_position_ids) + 1
        noleaf_index = torch.unique(mask_index).tolist()
        noleaf_num = len(noleaf_index) - 1
        leaf_num = self.router.total_tokens - noleaf_num

        retrieve_indices = torch.zeros(leaf_num, max_depth.item(), dtype=torch.long) - 1
        retrieve_indices = retrieve_indices.tolist()

        rid = 0
        position_ids_list = tree_position_ids.tolist()

        for i in range(self.router.total_tokens + 1):
            if i not in noleaf_index:
                cid = i
                depth = position_ids_list[i]
                for j in reversed(range(depth + 1)):
                    retrieve_indices[rid][j] = cid
                    cid = mask_index_list[cid - 1]
                rid += 1

        if logits_processor is not None:
            maxitem = self.router.total_tokens + 5

            def custom_sort(lst):
                # sort_keys=[len(list)]
                sort_keys = []
                for i in range(len(lst)):
                    sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
                return sort_keys

            retrieve_indices = sorted(retrieve_indices, key=custom_sort)

        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
        del mask_index, mask_index_list, noleaf_index, noleaf_num, leaf_num, max_depth, rid
        tree_position_ids = tree_position_ids.to(hidden_states.device)

        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids