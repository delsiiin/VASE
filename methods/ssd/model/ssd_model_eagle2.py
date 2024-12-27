import copy
import json
import time

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig,AutoConfig


from transformers import AutoTokenizer
import os
from huggingface_hub import hf_hub_download

from huggingface_hub import hf_hub_download

from .ssd_config import SSDConfig

# from transformers import LlamaForCausalLM

from .modeling_llama_ssd import LlamaForCausalLM as SSDLlamaForCausalLM

from .router import RouterModel

from .kv_cache import initialize_past_key_values, initialize_past_key_values_attn

from .utils_eagle import reset_ssd_mode_ea

import torch.nn.functional as F

import copy

class SSDModel(nn.Module):

    def __init__(
            self,
            base_model,
            ssd_config,
    ):

        super().__init__()
        self.base_model = base_model

        self.config = ssd_config

        self.topk = self.config.top_k_group
        
        self.router = RouterModel(self.config, self.config.attn_hid_dim)

        self.device = base_model.model.layers[-1].self_attn.q_proj.weight.device
       
        self.router.to(self.base_model.dtype).to(self.device)

        self.dtype = self.base_model.dtype

        # eagle modified
        self.top_k_draft = 6
        self.total_tokens = 20
        self.init_tree()
         # eagle modified

    @classmethod
    def from_pretrained(
            cls,
            base_model_path=None,
            ssd_model_path=None,
            Type="LLaMA",
            device=None,
            **kwargs,
    ):
        
        ssd_config = SSDConfig.from_pretrained(ssd_model_path)
            
        base_model = SSDLlamaForCausalLM.from_pretrained(
            base_model_path, **kwargs
        ).to(device)

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
            base_mode=None,
            hidden_states=None,
    ):
        
        if base_mode:
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
        
        else:

            all_draft_hidden_states = self.router(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values_attn,
                position_ids=position_ids,
                use_cache=use_cache
            )[0]

            all_draft_logits = self.base_model.lm_head(all_draft_hidden_states)

        if output_orig and base_mode:
            return outputs, orig, hidden_states
        return all_draft_logits
    
     # eagle modified
    def init_tree(self):
        self.tree_mask_init = torch.eye(self.top_k_draft, device=self.base_model.device)[None, None]
        self.tree_mask_init = self.tree_mask_init.to(self.base_model.device)
     # eagle modified


     # eagle modified
    def reset(self):
        self.router.ssd_mask = None
     # eagle modified

    def reset_kv(self):
        self.stable_kv_base = None
        self.stable_kv_attn = None

     # eagle modified
    def topK_generate(
            self, 
            attention_mask=None,
            labels=None,
            past_key_values=None,
            past_key_values_attn=None,
            output_orig=False,
            position_ids=None,
            use_cache=None,
            orig=None,
            hidden_states=None,
            logits_processor=None
        ):

        total_tokens = self.total_tokens
        top_k = self.top_k_draft

        # sample_token = input_ids[:, -1]

        scores_list = []
        parents_list = []
        ss_token = []

        # input_ids = input_ids[:, 1:]

        # self.reset()

        draft_logits = self(hidden_states = hidden_states, output_orig = True, base_mode = False, past_key_values_attn=past_key_values_attn)

        base_token = torch.argmax(orig[:, -1]).unsqueeze(0)

        # base_token_decoded = tokenizer.decode(base_token, skip_special_tokens=True)
        # print(f"Base token: {base_token_decoded}")

        # draft_logits.shape (3, 1, 66, 32000), logits.shape (1, 66, 32000)
        TOP_K = 20
        topk_all_val, topk_all_idx = torch.topk(draft_logits[...,-1,:], TOP_K, dim=-1) # idx是token val是概率
        # topk_all[1].shape (3, 1, 8)

        topk_all_idx = topk_all_idx.squeeze(1)
        topk_all_val = F.softmax(topk_all_val, dim=-1).squeeze(1)

        scores = topk_all_val[0] # (8)
        top = torch.topk(scores.view(-1), top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values
        scores = topk_p # (8)
        scores_list.append(scores) # (1, 8)
        ss_token.append(topk_all_idx[0][topk_index])
        # print(topk_all_idx[0], topk_index, ss_token)

        parents_list.append(torch.zeros(1, dtype=torch.long, device=scores.device)) # 索引
        
        # tree_mask = self.tree_mask_init
        topk_cs_index = torch.arange(top_k, device=scores.device) # 索引

        # self.topk -1
        for i in range(self.topk - 1):
            # self.router.ssd_mask = tree_mask

            # with Timer("sort1"):
            bias1 = top_k if i > 0 else 0
            bias2 = max(0, i-1)
            bias = 1 + top_k ** 2 * bias2 + bias1
            parents = (topk_cs_index + bias)
            parents_list.append(parents)

            top = torch.topk(topk_all_val[i+1].view(-1), top_k, dim=-1)
            topk_index, topk_p = top.indices, top.values
        
            cu_scores = scores[None] * topk_p[..., None] # (8, 8)
            # cu_scores = topk_p[None] * scores[..., None]
            # print(topk_p, scores)
            # print(cu_scores)

            topk_cs = torch.topk(cu_scores.view(-1), top_k, dim=-1)
            topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
      
            scores = topk_cs_p # (8)

            # print(topk_cs_index, i)

            # out_ids = topk_cs_index // top_k
            ss_token.append(torch.cat([topk_all_idx[i+1][topk_index]] * top_k, dim=0))
            # print(ss_token)
            cu_scores = cu_scores.view(-1)
            scores_list.append(cu_scores) # (1, 64)
            # tree_mask = torch.cat((tree_mask[:, :, out_ids], self.tree_mask_init), dim=3)

            # if self.threshold < 0 and cu_scores.max() < self.threshold:
            #     break
        
        # print(parents_list)
        # print(self.router.ssd_mask)

        # del parents_list,scores_list,ss_token
        # return draft_tokens, mask_index,tree_mask,tree_position_ids

        # with Timer("post"):

        scores_list = torch.cat(scores_list, dim=0).view(-1) # (64 * 2 + 8)

        ss_token_list = torch.cat(ss_token, dim=0).view(-1) # (64 * 2 + 8)
        
        top_scores = torch.topk(scores_list, total_tokens, dim=-1)
        top_scores_index = top_scores.indices # (63 total_tokens)
     
        top_scores_index = torch.sort(top_scores_index).values

        # print(ss_token_list)

        draft_tokens = ss_token_list[top_scores_index] # (63 total_tokens)
        draft_tokens = torch.cat((base_token, draft_tokens), dim=0)
        
        # draft_tokens = torch.cat((sample_token, draft_tokens), dim=0)

        # print(draft_tokens)

        draft_parents = torch.cat(parents_list, dim=0)[top_scores_index // top_k].long() # (63 total_tokens)

        # print(draft_parents)
    
        mask_index = torch.searchsorted(top_scores_index, draft_parents - 1, right=False)
        # mask_index[(top_scores_index[mask_index]!=draft_parents - 1)]=-1
        mask_index[draft_parents == 0] = -1
        mask_index = mask_index + 1
        mask_index_list = mask_index.tolist()
       
        # with Timer("mask"):
        tree_mask = torch.eye(total_tokens + 1).bool()
        tree_mask[:, 0] = True
        for i in range(total_tokens):
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
        leaf_num = total_tokens - noleaf_num

        retrieve_indices = torch.zeros(leaf_num, max_depth.item(), dtype=torch.long) - 1
        retrieve_indices = retrieve_indices.tolist()

        rid = 0
        position_ids_list = tree_position_ids.tolist()

        for i in range(total_tokens + 1):
            if i not in noleaf_index:
                cid = i
                depth = position_ids_list[i]
                for j in reversed(range(depth + 1)):
                    retrieve_indices[rid][j] = cid
                    cid = mask_index_list[cid - 1]
                rid += 1

        if logits_processor is not None:
            maxitem = total_tokens + 5

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

     # eagle modified
