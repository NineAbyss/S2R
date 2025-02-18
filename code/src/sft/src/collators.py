import torch
from typing import List, Dict, Callable
from transformers import PreTrainedTokenizer
from arguments import CustomTrainingArguments, SFTTrainingArguments, SFTWeightedTrainingArugments, OfflineWeightedPolicyTrainingArguments
from copy import deepcopy
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from utils import print_object_on_main_process


def _llm_tokenize(prompts: List[str], texts: List[str], tokenizer: PreTrainedTokenizer, args: CustomTrainingArguments) -> Dict[str, torch.Tensor]:
    input_ids = []
    labels = []
    for prompt, text in zip(prompts, texts):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        text_ids = tokenizer.encode(text, add_special_tokens=False)
        response_start_idx = len(prompt_ids)
        if prompt_ids != text_ids[:response_start_idx]:
            response_start_idx -= 1
        if args.add_special_tokens:
            response_start_idx += 1
            text_ids = [tokenizer.bos_token_id] + text_ids + [tokenizer.eos_token_id]

        label = deepcopy(text_ids)
        if args.only_predict_answer:
            # print_object_on_main_process('label', len(label))
            label[:response_start_idx] = [-100] * response_start_idx
            # print_object_on_main_process('label', len(label))

        if len(text_ids) > args.model_max_length:
            text_ids = text_ids[-args.model_max_length:]
            label = label[-args.model_max_length:]

        input_ids.append(torch.tensor(text_ids))
        labels.append(torch.tensor(label))
    
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    if args.pad_labels_with_ignore:
 
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    else:
        labels = pad_sequence(labels, batch_first=True, padding_value=tokenizer.pad_token_id)

    if args.debug_mode: # If debug_model is True, then pad the sequence into the max token length.
        input_ids = F.pad(input_ids, (0, tokenizer.model_max_length - input_ids.shape[0]), mode='constant', value=tokenizer.pad_token_id)
        labels = F.pad(labels, (0, tokenizer.model_max_length - input_ids.shape[0]), mode='constant', value=args.ignore_token_id 
                    if args.pad_labels_with_ignore else tokenizer.pad_token_id)

    attention_mask = torch.ne(input_ids, tokenizer.pad_token_id)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

    
def sft_data_collator(tokenizer: PreTrainedTokenizer, args: SFTTrainingArguments) -> Callable:
    def collactor(examples) -> Dict[str, torch.Tensor]:
        texts = []
        prompts = []
        for example in examples:
            text = example['prompt'] + example['answer']
            texts.append(text)
            prompts.append(example['prompt'])

        return _llm_tokenize(prompts, texts, tokenizer, args)
    
    return collactor

    
def sft_weighted_data_collator(tokenizer: PreTrainedTokenizer, args: SFTWeightedTrainingArugments) -> Callable:
    def collactor(examples) -> Dict[str, torch.Tensor]:
        texts = []
        prompts = []
        weights = []
        for example in examples:
            text = example['prompt'] + example['answer']
            texts.append(text)
            prompts.append(example['prompt'])
            weights.append(example.get('weight', 1))
        ret = _llm_tokenize(prompts, texts, tokenizer, args)
        ret.update({"weights": torch.tensor(weights).float()})

        return ret

    return collactor


def offline_weighted_policy_data_collator(tokenizer: PreTrainedTokenizer, args: OfflineWeightedPolicyTrainingArguments):
    _sft_weighted_data_collator = sft_weighted_data_collator(tokenizer, args)
    def collator(examples) -> Dict[str, torch.Tensor]:
        ret = _sft_weighted_data_collator(examples)
        # print_object_on_main_process('input_ids', ret['input_ids'].shape)
        # print_object_on_main_process('labels', ret['labels'].shape)
        rewards = torch.tensor(
            [example['reward'] for example in examples]
        )
        sft_mask = torch.tensor([ 1. if example.get('type', 'sample') == 'sft' else 0. for example in examples])
        ret.update({"rewards": rewards, "sft_mask": sft_mask})
        return ret
    
    return collator