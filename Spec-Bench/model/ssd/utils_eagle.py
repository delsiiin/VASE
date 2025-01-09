import copy
import random

# typing 
from typing import List, Tuple
import time
import torch

# TODO
# from transformers import LlamaTokenizer
# tokenizer=LlamaTokenizer.from_pretrained("/home/lyh/weights/hf/vicuna_v13/7B/")

TOPK = 10  # topk for sparse tree

from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)


class Timer:
    def __init__(self,name):
        self.name = name
    def __enter__(self):
        torch.cuda.synchronize()
        self.start = time.perf_counter()


    def __exit__(self, exc_type, exc_value, traceback):
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.start
        print(f'{self.name} took {elapsed} seconds')


def prepare_logits_processor(
        temperature: float = 0.0,
        repetition_penalty: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    if temperature > 1e-5:
        if temperature >= 1e-5 and temperature != 1.0:
            processor_list.append(TemperatureLogitsWarper(temperature))
        if repetition_penalty > 1.0:
            processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
        if 1e-8 <= top_p < 1.0:
            processor_list.append(TopPLogitsWarper(top_p))
        if top_k > 0:
            processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


# test_processor = prepare_logits_processor(
#         0.0, 0.0, -1, 1
#     )


def pad_path(path: List[int], length: int, pad_value: int = -2) -> List[int]:
    """
    Pad the given path list with a specific value up to a specified length.

    Parameters:
    - path (list): The original list that needs padding.
    - length (int): The desired length of the padded list.
    - pad_value (optional, default=-2): The value to use for padding.

    Returns:
    - list: A new list based on the original path but padded to the desired length.

    Example:
    >>> pad_path([1,2,3], 5)
    [1, 2, 3, -2, -2]

    Note:
    If the given path is already longer than the specified length,
    then no padding occurs, and the original path is returned.
    """

    # Calculate the number of padding values needed by subtracting the length
    # of the path from the desired length.
    # Append the padding values to the original path and return the new list.
    return path + [pad_value] * (length - len(path))

def initialize_ssd_ea_hid(input_ids, model, logits_processor):

    _, orig, hidden_states = model.tree_forward(input_ids, output_orig = True, past_key_values=model.past_key_values)
    
    draft_tokens, retrieve_indices,tree_mask,tree_position_ids = model.topK_generate(hidden_states=hidden_states, past_key_values_attn=model.past_key_values_attn, orig=orig, logits_processor=logits_processor)

    return draft_tokens, retrieve_indices,tree_mask,tree_position_ids, orig, hidden_states


def reset_ssd_mode_ea(
    model, attn
):
    model.base_model.model.ssd_mask = None
    model.base_model.model.ssd_mode = None

    if attn:
        model.router.ssd_mask = None


def reset_past_key_values(passed_key_values: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Resets the current lengths in the passed key-values to zero.

    This function is designed to be used during the evaluation of a baseline model.
    It iterates through each layer's key-values and sets their current lengths to zero,
    effectively resetting their state.

    Args:
    - passed_key_values (list of torch.Tensor): Contains past hidden states and past attention values for each layer.

    Returns:
    - passed_key_values (list of torch.Tensor): Updated past hidden states and past attention values with reset lengths.
    """
    for i in range(len(passed_key_values)):
        for j in range(2):
            passed_key_values[i][j].current_length.fill_(0)
    return passed_key_values

def ssd_tree_decoding_ea_hid(
        model,
        tree_candidates,
        tree_position_ids,
        input_ids,
        retrieve_indices,
):
    position_ids = tree_position_ids + input_ids.shape[1]

    _, tree_logits, hidden_states = model.tree_forward(
        tree_candidates,
        output_orig=True,
        past_key_values=model.past_key_values,
        position_ids=position_ids,
    )
    
    logits = tree_logits[0, retrieve_indices]

    return logits, hidden_states

def evaluate_posterior_ea(
        logits: torch.Tensor,
        candidates: torch.Tensor,
        logits_processor=None,
):
    """
    Evaluate the posterior probabilities of the candidates based on the provided logits and choose the best candidate.

    Depending on the temperature value, the function either uses greedy decoding or evaluates posterior
    probabilities to select the best candidate.

    Args:
    - logits (torch.Tensor): Predicted logits of shape (batch_size, sequence_length, vocab_size).
    - candidates (torch.Tensor): Candidate token sequences.
    - temperature (float): Softmax temperature for probability scaling. A value of 0 indicates greedy decoding.
    - posterior_threshold (float): Threshold for posterior probability.
    - posterior_alpha (float): Scaling factor for the threshold.

    Returns:
    - best_candidate (torch.Tensor): Index of the chosen best candidate.
    - accept_length (int): Length of the accepted candidate sequence.
    """
    # Greedy decoding based on temperature value
    if logits_processor is None:
        # Find the tokens that match the maximum logits for each position in the sequence
        posterior_mask = (
                candidates[:, 1:].to(logits.device) == torch.argmax(logits[:, :-1], dim=-1)
        ).int()
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()
        # Choose the best candidate
        if accept_length == 0:
            # Default to the first candidate if none are accepted
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        return best_candidate, accept_length, logits[best_candidate, accept_length][None, None]

    else:
        accept_length = 1
        accept_cand = candidates[0][:1]
        best_candidate = 0
        for i in range(1, candidates.shape[1]):
            if i != accept_length:
                break
            adjustflag = False
            is_eq = (candidates[:, :accept_length] == accept_cand).all(dim=1)
            fi = torch.nonzero(is_eq, as_tuple=True)[0][0]
            gt_logits = logits[fi, i - 1][None]
            gt_logits = logits_processor(None, gt_logits)[0]
            gtp = torch.softmax(gt_logits, dim=0)
            candidates_set = []
            for j in range(candidates.shape[0]):
                if is_eq[j]:
                    x = candidates[j, i]
                    xi = x.item()
                    if xi in candidates_set or xi == -1:
                        continue
                    candidates_set.append(xi)
                    r = random.random()
                    px = gtp[xi]
                    qx = 1.0
                    acp = px / qx
                    if r <= acp:
                        accept_cand = torch.cat((accept_cand, x[None]), dim=0)
                        accept_length += 1
                        best_candidate = j
                        break
                    else:
                        gtp[xi] = 0
                        gtp = gtp / gtp.sum()
                        adjustflag = True
        if adjustflag and accept_length != candidates.shape[1]:
            sample_p = gtp
        else:
            gt_logits = logits[best_candidate, accept_length - 1]
            sample_p = torch.softmax(gt_logits, dim=0)
        return torch.tensor(best_candidate), accept_length - 1, sample_p[None, None]

@torch.no_grad()
def update_inference_inputs_ssd_ea_hid(
        input_ids,
        candidates,
        best_candidate,
        accept_length,
        retrieve_indices,
        new_token,
        model,
        hidden_states_new,
        orig,
        logits_processor,
):

    prev_input_len = input_ids.shape[1]
    # Map the best candidate indices to the original indices in the sequence
    select_indices = (
            retrieve_indices[best_candidate, : accept_length + 1] + prev_input_len
    )

    # Append the tokens from the best candidate to the input sequence
    input_ids = torch.cat(
        [input_ids, candidates[None, best_candidate, : accept_length + 1].to(input_ids.device)], dim=-1
    )
    
    # Update the past key values based on the selected tokens
    # Source tensor that contains relevant past information based on the selected candidate
    for past_key_values_data in model.past_key_values_data:
        tgt = past_key_values_data[..., select_indices.to(past_key_values_data.device), :]
        # Destination tensor where the relevant past information will be stored
        dst = past_key_values_data[..., prev_input_len: prev_input_len + tgt.shape[-2], :]
        # Copy relevant past information from the source to the destination
        dst.copy_(tgt, non_blocking=True)

    # Update the current length tensor (currently only support batch size is 1)
    model.current_length_data.fill_(prev_input_len + tgt.shape[-2])

    # Update the past key values based on the selected tokens
    # Source tensor that contains relevant past information based on the selected candidate
    for past_key_values_data_attn in model.past_key_values_data_attn:
            tgt_attn = past_key_values_data_attn[..., select_indices.to(past_key_values_data_attn.device), :]
            # Destination tensor where the relevant past information will be stored
            dst_attn = past_key_values_data_attn[..., prev_input_len: prev_input_len + tgt_attn.shape[-2], :]
            # Copy relevant past information from the source to the destination
            dst_attn.copy_(tgt_attn, non_blocking=True)

    # Update the current length tensor (currently only support batch size is 1)
    model.current_length_data_attn.fill_(prev_input_len + tgt_attn.shape[-2])

    retrieve_hidden_state_new = hidden_states_new[:, retrieve_indices]
    accept_hidden_state_new = retrieve_hidden_state_new[:, best_candidate, : accept_length + 1]

    # hidden_state = torch.cat((hidden_state, accept_hidden_state_new), dim=1)

    # draft_tokens, retrieve_indices,tree_mask,tree_position_ids,orig = model.topK_generate(input_ids=input_ids, past_key_values=None, past_key_values_attn=None, tokenizer=tokenizer)

    model.base_model.model.ssd_mask = None
    model.router.ssd_mask = None

    draft_tokens, retrieve_indices,tree_mask,tree_position_ids = model.topK_generate(hidden_states=accept_hidden_state_new, orig=orig, past_key_values_attn=model.past_key_values_attn, logits_processor=logits_processor)

    new_token += accept_length + 1

    return input_ids, draft_tokens, retrieve_indices,tree_mask,tree_position_ids, new_token


if __name__ == "__main__":
    logits = torch.randn(1, 5)
    tp = prepare_logits_processor(0.9, 0, 0.9, 0)
    l = tp(None, logits)
    if tp is None:
        print(tp)