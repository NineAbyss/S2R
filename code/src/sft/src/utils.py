import torch
import json
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from datasets import Dataset
from transformers import TrainingArguments, PreTrainedModel, PreTrainedTokenizer
import os
    
    
def is_main_process():
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            return True
        else:
            return False
    else:
        return True


def print_rank_0(message, end='\n', color='green') -> None:
    if color == 'default':
        prefix = "\033[38m"
    elif color == 'red':
        prefix = "\033[31m"
    elif color == 'green':
        prefix = "\033[32m"
    elif color == 'yellow':
        prefix = "\033[33m"
    elif color == 'blue':
        prefix = "\033[34m"
    elif color == 'pink':
        prefix = "\033[35m"
    elif color == 'cyan':
        prefix = "\033[36m"

    postfix="\033[0m"
    if is_main_process():
        print(prefix + repr(message) + postfix, flush=True, end=end)


def print_object_on_main_process(name: str, obj: object, split_line_color="yellow", object_color="pink") -> None:
    print_rank_0(">"*30 + name, color=split_line_color)
    print_rank_0(obj, color=object_color)
    print_rank_0(">"*30, color=split_line_color)


def read_json_or_jsonl_data(data_path: str) -> List:
    if data_path.endswith('json'):
        with open(data_path, 'r') as f:
            data_list = json.load(f)
    elif data_path.endswith('jsonl'):
        with open(data_path, 'r') as f:
            lines = f.read().strip().split('\n')
            data_list = [json.loads(l) for l in lines]
    else:
        raise ValueError("The data file must end with json or jsonl.")
    
    print_rank_0(f">>> totally load {len(data_list)} data from {data_path}.")
    return data_list

    
def load_data_from_paths(data_paths: List[str]) -> List[Dict[str, Any]]:
    total_data_list = []
    i = 0
    for data_path in data_paths:
        data_list = read_json_or_jsonl_data(data_path)
        for data in tqdm(data_list, disable=not is_main_process()):
            data['id'] = i
            i += 1
            total_data_list.append(data)
    print_rank_0(f">>> total load {len(total_data_list)} data.")
    return total_data_list

    
def getDataset(args: TrainingArguments, data_transform: Callable, type='train') -> Union[Dataset, Dict[str, Dataset]]:
    if type == 'train':
        if args.data_paths is None and args.data_dir is None:
            return None
        if args.data_paths is not None:
            data_paths = args.data_paths
        else:
            data_paths = [os.path.join(args.data_dir, path) for path in os.listdir(args.data_dir)]
        print_rank_0(data_paths)
        data_list = data_transform(load_data_from_paths(data_paths), args)
        return Dataset.from_list(data_list)
    else:
        eval_dataset = {}
        if args.eval_data_paths is None and args.eval_data_dir is None:
            return None
        if args.eval_data_paths is not None:
            data_paths = args.eval_data_paths
        else:
            data_paths = [os.path.join(args.eval_data_dir, path) for path in os.listdir(args.eval_data_dir)]       
        if args.eval_dataset_merge_mode in ['separate', 'both']:
            if args.eval_dataset_merge_mode == 'both':
                eval_dataset['all'] = []
            for path in data_paths:
                sub_data_list = data_transform(load_data_from_paths([path]), args)
                if args.eval_dataset_merge_mode == 'both':
                    eval_dataset['all'].extend(sub_data_list)
                _, name = os.path.split(path)
                eval_dataset[name] = Dataset.from_list(sub_data_list)
            if args.eval_dataset_merge_mode == 'both':
                eval_dataset['all'] = Dataset.from_list(eval_dataset['all'])

        elif args.eval_dataset_merge_mode == 'merge':
            eval_dataset = data_transform(load_data_from_paths(data_paths), args)
            eval_dataset = Dataset.from_list(eval_dataset)
        return eval_dataset

        
def set_special_tokens(tokenizer: PreTrainedTokenizer, model: PreTrainedModel) -> None:
    DEFAULT_PAD_TOKEN = "<pad>"
    DEFAULT_EOS_TOKEN = "</s>"
    DEFAULT_BOS_TOKEN = "<s>"
    DEFAULT_UNK_TOKEN = "<unk>"
    
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict['bos_token'] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings: torch.Tensor = model.get_input_embeddings().weight.data
        output_embeddings: torch.Tensor = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg