import re
import copy
import json
import torch
import random
import transformers
import numpy as np
from tqdm import tqdm
from pathlib import Path

from dataclasses import dataclass
from torch.utils.data import Dataset
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, Sequence, Union, Tuple, List

from utils import jload, print_rank_0, print_rank_0, get_rewards


IGNORE_INDEX = -100

def get_instance_reward_from_reward_dist(reward_dist: str) -> float:
    for i in range(len(reward_dist) - 1, -1, -1):
        if reward_dist[i] == 'R':
            return 1.0 if reward_dist[i+1]=="+" else -1.0


def compute_baseline(reward_history: List, use_softmax_norm:str = False, tau=1.0, min_group_baseline:int=10) -> float:
    """
    use Softmax norm to cumpute baseline like kimi;
    
    :param reward_history: list of float
    :param tau: float
    :return: float
    """

    if len(reward_history) > 0:
        if use_softmax_norm:
            baseline = tau * np.log(np.mean(np.exp(np.array(reward_history) / tau))) \
                if len(reward_history) >= min_group_baseline else 0.0
        else:
            baseline = np.mean(reward_history).item() \
                if len(reward_history) >= min_group_baseline else 0.0
    else:
        baseline = 0.0

    return baseline



def _tokenize_fn(
    strings: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]

    tokenized_dict = {
        "input_ids": input_ids,
        "labels": labels,
        "input_ids_lens": input_ids_lens,
        "labels_lens": labels_lens,
    }

    return tokenized_dict


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX

    preprocess_dict = {
        "input_ids": input_ids,
        "labels": labels,
    }
    return preprocess_dict


class OfflineRLDataset(Dataset):
    """Dataset for offline RL."""

    def __init__(
        self,
        data_path: Union[str, List[str]],
        tokenizer: transformers.PreTrainedTokenizer,
        debug_mode: bool = False,
        precompute_reward: bool = True,
        reward_load_path: str = None,
        reward_save_path: str = None,
        reward_delay_factor: float = 1.0,
        use_bonus: bool = False,
        use_veri_bonus: bool = False,
        use_prefix_baseline: bool = False,
        use_position_baseline: bool = False,
        use_level_baseline: bool = False,
        use_accuracy_baseline: bool = False,
        use_softmax_norm: bool = False,
        min_group_baseline: int = 10,
        use_process_rl: bool = True,
        **kwargs
    ):
        super()
        print_rank_0("Loading data...")
        
        self.reward_delay_factor = reward_delay_factor

        list_data_dict = []
        if isinstance(data_path, list):
            for path in data_path:
                new_data = jload(path)
                if new_data and 'reward' not in new_data[0]:
                    if list_data_dict:
                        new_data = random.sample(new_data, min(len(list_data_dict), len(new_data)))
                list_data_dict.extend(new_data)
        else:
            list_data_dict = jload(data_path)

        if debug_mode:
            list_data_dict = list_data_dict[:100]

        print_rank_0("Length of training data: " + str(len(list_data_dict)))

        # Step 1: Precompute rewards
        if precompute_reward:

            print_rank_0("Precomputing rewards...")
            print_rank_0("Computing baseline...")

            self._initialize_baseline(
                use_accuracy=use_accuracy_baseline,
                use_level=use_level_baseline,
                use_process_rl=use_process_rl
            )

            self._compute_baseline(
                dataset=list_data_dict,
                use_prefix_baseline=use_prefix_baseline,
                use_position_baseline=use_position_baseline,
                use_level_baseline=use_level_baseline,
                use_accuracy_baseline=use_accuracy_baseline,
                use_softmax_norm=use_softmax_norm,
                min_group_baseline=min_group_baseline,
                use_process_rl=use_process_rl
            )

            print_rank_0("Computing reward...")

            if reward_load_path:
                list_data_dict_ = jload(reward_load_path)
                if len(list_data_dict_) != len(list_data_dict):
                    list_data_dict_, misalign_idxs = self.precompute_reward(
                        list_data_dict,
                        tokenizer,
                        fp_save_path=reward_save_path,
                        use_bonus=use_bonus,
                        use_veri_bonus=use_veri_bonus,
                        use_prefix_baseline=use_prefix_baseline,
                        use_position_baseline=use_position_baseline,
                        use_level_baseline=use_level_baseline,
                        use_accuracy_baseline=use_accuracy_baseline,
                        use_process_rl=use_process_rl
                    )
                list_data_dict = list_data_dict_
            else:
                list_data_dict, misalign_idxs = self.precompute_reward(
                    list_data_dict,
                    tokenizer,
                    fp_save_path=reward_save_path,
                    use_bonus=use_bonus,
                    use_veri_bonus=use_veri_bonus,
                    use_prefix_baseline=use_prefix_baseline,
                    use_position_baseline=use_position_baseline,
                    use_level_baseline=use_level_baseline,
                    use_accuracy_baseline=use_accuracy_baseline,
                    use_process_rl=use_process_rl
                )

        # Step 2: Tokenize the input and output
        # Step 1: Format input and output based on model format
        print_rank_0("Formatting inputs...")
        sources = [d["prompt"] for idx, d in enumerate(list_data_dict)]
        targets = [d["answer"] for idx, d in enumerate(list_data_dict)]

        print_rank_0("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        # Step 4: Gather all data
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.gather_dict(list_data_dict)

        assert len(self.rewards) == len(self.weights) == len(self.input_ids) == len(self.labels) == len(self.sft_masks)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        get_dict = {
            "input_ids": self.input_ids[i],
            "labels": self.labels[i],
            "reward": self.rewards[i],
            "weight": self.weights[i],
            "token_rewards": self.token_rewards[i],
            "answer_masks": self.answer_masks[i],
            "part_idx_masks": self.part_idx_masks[i],
            "round1_answer_masks": self.round1_answer_masks[i],
            "split_token_masks": self.split_token_masks[i],
            "round2toN_answer_masks": self.round2toN_answer_masks[i],
            "sft_mask": self.sft_masks[i],
            "veri_rewards": self.veri_rewards[i],
            "veri_answers": self.veri_answers[i],
            "response_extracted_answers": self.response_extracted_answers[i],
            "response_rewards": self.response_rewards[i],
            "gold_extracted_answer": self.gold_extracted_answers[i]
        }
        return get_dict
    

    def gather_dict(self, list_data_dict:List):
        self.rewards = [example.get("reward", 1.0) for example in list_data_dict]
        self.weights = [example.get("weight", 1.0) for example in list_data_dict]
        self.sft_masks = [0. if 'reward' in example else 1. for example in list_data_dict]

        self.token_rewards = [example.get("token_rewards", []) for example in list_data_dict]
        self.answer_masks = [example.get("answer_masks", []) for example in list_data_dict]
        self.part_idx_masks = [example.get("part_idx_masks", []) for example in list_data_dict]
        self.round1_answer_masks = [example.get("round1_answer_masks", []) for example in list_data_dict]
        self.split_token_masks = [example.get("split_token_masks", []) for example in list_data_dict]
        self.round2toN_answer_masks = [example.get("round2toN_answer_masks", []) for example in list_data_dict]

        self.veri_rewards = [example.get("veri_rewards", []) for example in list_data_dict]
        self.veri_answers = [example.get("veri_answers", []) for example in list_data_dict]

        self.gold_extracted_answers = [example.get("gold_extracted_answer", "") for example in list_data_dict]
        self.response_extracted_answers = [example.get("response_extracted_answers", []) for example in list_data_dict]
        self.response_rewards = [example.get("response_rewards", []) for example in list_data_dict]

        assert len(self.rewards) == len(self.weights) == len(self.sft_masks) == len(self.token_rewards) == len(self.answer_masks) == len(self.part_idx_masks) == len(self.round1_answer_masks) == len(self.split_token_masks) == len(self.round2toN_answer_masks) == len(self.veri_rewards) == len(self.veri_answers) == len(self.gold_extracted_answers) == len(self.response_extracted_answers) == len(self.response_rewards)
    

    def _initialize_baseline(self, use_level:bool=False, use_accuracy:bool=False, use_process_rl=True):
        if use_process_rl:
            if use_level or use_accuracy:
                self.global_baseline_dict = defaultdict(lambda: defaultdict(list))
                self.global_baselines = defaultdict(lambda: defaultdict(float))

            else:
                self.global_baseline_dict = defaultdict(list)
                self.global_baselines = defaultdict(float)

        else:
            if use_level or use_accuracy:
                self.global_baseline_dict = defaultdict(list)
                self.global_baselines = defaultdict(float)
            else:
                self.global_baselines = []


    def _compute_baseline(self, dataset: List[Dict],
                        use_prefix_baseline: bool = False,
                        use_position_baseline: bool = False,
                        use_level_baseline: bool = False, 
                        use_accuracy_baseline: bool = False,
                        use_softmax_norm:bool=False,
                        min_group_baseline:int=10,
                        use_process_rl:bool=True):

        for sample in dataset:
            
            accuracy = sample["accuracy"]
            level = sample["level"]
            reward_dist = sample["reward_dist"]

            prefix = ""
            if use_process_rl:

                for i in range(2, len(reward_dist) + 1, 2):
                    
                    current_reward_dist = reward_dist[:i]
                    reward_ = 1.0 if current_reward_dist[-1]=="+" else -1.0

                    if use_prefix_baseline and use_accuracy_baseline:
                        self.global_baseline_dict[accuracy][prefix].append(reward_)
                        # update prefix
                        prefix = current_reward_dist

                    elif use_prefix_baseline:
                        self.global_baseline_dict[prefix].append(reward_)
                        # update prefix
                        prefix = current_reward_dist

                    elif use_position_baseline:
                        pos = i // 2 # start from 1
                        self.global_baseline_dict[pos].append(reward_)

                    elif use_level_baseline:
                        pos = i // 2
                        self.global_baseline_dict[level][pos].append(reward_)
                            
                    elif use_accuracy_baseline:
                        pos = i // 2
                        self.global_baseline_dict[accuracy][pos].append(reward_)

            else:
                instance_reward = get_instance_reward_from_reward_dist(reward_dist)
                if use_level_baseline or use_accuracy_baseline:
                    self.global_baseline_dict[level].append(instance_reward)
                    self.global_baseline_dict[accuracy].append(instance_reward)
                else:
                    self.global_baselines.append(instance_reward)
        
        if (use_level_baseline or use_accuracy_baseline) and use_process_rl:
            for key, value in self.global_baseline_dict.items():
                for k, v in value.items():
                    self.global_baselines[key][k] = compute_baseline(v, use_softmax_norm=use_softmax_norm, min_group_baseline=min_group_baseline)
        elif use_process_rl or use_level_baseline or use_accuracy_baseline:
            for key, value in self.global_baseline_dict.items():
                self.global_baselines[key] = compute_baseline(value, use_softmax_norm=use_softmax_norm, min_group_baseline=min_group_baseline)
        else:
            self.global_baselines = sum(self.global_baselines) / len(self.global_baselines)


        print_rank_0("Baseline has computed.")


    def compute_advantage(
        self,
        reward:float = None, 
        level:int = None,
        accuracy:float = None,
        current_part_idx:int = None,
        current_prefix: str = None,
        use_prefix_baseline: bool = False,
        use_position_baseline: bool = False,
        use_level_baseline: bool = False,
        use_accuracy_baseline: bool = False,
        use_process_rl: bool = True
    ) -> float:
        if use_process_rl:
            if use_prefix_baseline and use_accuracy_baseline:
                return reward - self.global_baselines[accuracy][current_prefix]
            elif use_prefix_baseline:
                return reward - self.global_baselines[current_prefix]
            elif use_position_baseline:
                return reward - self.global_baselines[current_part_idx]
            elif use_level_baseline:
                return reward - self.global_baselines[level][current_part_idx]
            elif use_accuracy_baseline:
                return reward - self.global_baselines[accuracy][current_part_idx]
            else:
                return reward
            
        else:
            if use_level_baseline:
                return reward - self.global_baselines[level]
            elif use_accuracy_baseline:
                return reward - self.global_baselines[accuracy]
            else:
                return reward - self.global_baselines

    def precompute_reward(
        self,
        dataset: List[Dict],
        tokenizer: transformers.PreTrainedTokenizer,
        use_bonus: bool = False,
        use_veri_bonus: bool = False,
        use_prefix_baseline: bool = False,
        use_position_baseline: bool = False,
        use_level_baseline: bool = False,
        use_accuracy_baseline: bool = False,
        fp_save_path: str = None,
        use_process_rl: bool = True
    ) -> Tuple[List[Dict], List]:

        eos_token_id_str = str(tokenizer.eos_token_id)
        str_check_tokens = [" Wait,", "Wait,", "Let me recheck my solution.\n\n", " Let me recheck my solution.\n\n"]
        str_check_token_ids = [
            "_".join([str(token) for token in tokenizer(token, add_special_tokens=False).input_ids]) for token in str_check_tokens
        ]
        str_retry_token_ids = "_".join([str(token) for token in tokenizer("Let me try again.\n\n", add_special_tokens=False).input_ids])
        str_retry_token_ids_other = "_".join([str(token) for token in tokenizer(" Let me try again.\n\n", add_special_tokens=False).input_ids])

        updated_dataset = []

        group_reward_dict = {}

        misaligned_idx_list = []

        for idx, example in tqdm(enumerate(dataset)):
            prompt = example["prompt"]
            answer = example["answer"]
            level = example["level"]
            accuracy = example["accuracy"]
            reward_dist = example["reward_dist"]

            gold_solution = example.get("solution", "")
            gold_extracted_answer = example.get("gold_extracted_answer", "")

            prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=False)
            input_length = len(prompt_token_ids)
            response_token_ids = tokenizer.encode(answer, add_special_tokens=False)

            str_response_token_ids = "_".join(str(tk) for tk in response_token_ids)

            if eos_token_id_str in str_response_token_ids:
                str_response_token_ids = str_response_token_ids.split(eos_token_id_str)[0]
            str_response_token_ids = str_response_token_ids.strip("_")

            if "deepseek" in getattr(tokenizer, "name_or_path", ""):
                str_check_token_ids = [
                    str_check_ids.replace("100000_", "") for str_check_ids in str_check_token_ids
                ]
                str_retry_token_ids = str_retry_token_ids.replace("100000_", "")
                str_retry_token_ids_other = str_retry_token_ids_other.replace("100000_", "")

            str_check_token_ids_pattern = "|".join(map(re.escape, str_check_token_ids))

            split_texts = re.split(f"({str_check_token_ids_pattern}|{str_retry_token_ids_other}|{str_retry_token_ids})", str_response_token_ids)

            full_ids = prompt_token_ids[:]
            token_rewards = [0] * input_length
            answer_masks = [0] * input_length
            round1_answer_masks = [0] * input_length 
            split_token_masks = [0] * input_length
            round2toN_answer_masks = [0] * input_length
            part_idx_masks = [0] * input_length
            response_extracted_answers = []

            round1_res_ids_str = split_texts[0] if len(split_texts) > 0 else ""
            if round1_res_ids_str.strip("_"):
                round1_tokenized = [int(token) for token in round1_res_ids_str.strip("_").split("_") if token]
            else:
                round1_tokenized = []

            round1_res_text = tokenizer.decode(round1_tokenized)
            r_, _, cg_ = get_rewards([gold_solution], [gold_extracted_answer], [round1_res_text], strict=False)
            r_, cg_ = r_[0], cg_[0]

            current_part_idx = 1
            current_last_reward = r_  
            current_reward_dist = ""

            current_reward = self.compute_advantage(
                        reward=r_,
                        level=level,
                        accuracy=accuracy,
                        current_part_idx=current_part_idx,
                        current_prefix=current_reward_dist,
                        use_prefix_baseline=use_prefix_baseline,
                        use_position_baseline=use_position_baseline,
                        use_level_baseline=use_level_baseline,
                        use_accuracy_baseline=use_accuracy_baseline,
                        use_process_rl=use_process_rl
                    )

            current_reward_dist += "R+" if r_ > 0 else "R-" 

            if use_bonus:
                if r_ > 0:
                    current_reward = current_reward * 1
                else:
                    current_reward = current_reward
            
            round1_answer_masks += [1] * len(round1_tokenized)
            round1_answer_masks = round1_answer_masks + [0] * (len(response_token_ids) - len(round1_tokenized))

            full_ids += round1_tokenized
            answer_masks += [1] * len(round1_tokenized)
            token_rewards += [current_reward] * len(round1_tokenized)

            average_part_reward = sum([current_reward] * len(round1_tokenized)) / len(round1_tokenized)

            part_idx_masks += [1] * len(round1_tokenized)
            split_token_masks += [0] * len(round1_tokenized)
            round2toN_answer_masks += [0] * len(round1_tokenized)
            response_extracted_answers.append(cg_)
            
            last_split_token = None

            veri_rewards = []
            veri_answers = []
            response_rewards = [current_reward]
            final_rewards_for_logging = [current_reward]

            for i, subtext_ids_str in enumerate(split_texts[1:], start=1):

                subtext_ids_str = subtext_ids_str.strip("_")
                if not subtext_ids_str:
                    continue

                if subtext_ids_str in str_check_token_ids \
                   or subtext_ids_str == str_retry_token_ids \
                   or subtext_ids_str == str_retry_token_ids_other:

                    subtext_ids = [int(x) for x in subtext_ids_str.split("_") if x]
                    
                    full_ids += subtext_ids
                    token_rewards += [0] * len(subtext_ids)
                    answer_masks += [1] * len(subtext_ids)
                    split_token_masks += [1] * len(subtext_ids)
                    round2toN_answer_masks += [0] * len(subtext_ids)
                    part_idx_masks += [0] * len(subtext_ids)
                                
                    last_split_token = subtext_ids_str
                    continue

                if last_split_token in str_check_token_ids:
                    
                    current_part_idx += 1

                    current_tokenized = [int(x) for x in subtext_ids_str.split("_") if x]
                    subtext = tokenizer.decode(current_tokenized)

                    if "is incorrect" in subtext or "cannot verify" in subtext:
                        veri_answer = "incorrect"
                    elif "is correct" in subtext or "verified" in subtext or\
                         "is indeed" in subtext or "verified to be correct" in subtext or\
                         "verified as correct" in subtext:
                        veri_answer = "correct"
                    else:
                        if i != len(split_texts) - 2:
                            veri_answer = "incorrect"
                        else:
                            veri_answer = "correct"

                    if veri_answer == "":
                        veri_reward_ = 0

                    elif (current_last_reward > 0 and veri_answer == "correct") or\
                         (current_last_reward < 0 and veri_answer == "incorrect"):
                        veri_reward_ = 1
                    else:
                        veri_reward_ = -1

                    veri_reward = self.compute_advantage(
                        reward=veri_reward_,
                        level=level,
                        accuracy=accuracy,
                        current_part_idx=current_part_idx,
                        current_prefix=current_reward_dist,
                        use_prefix_baseline=use_prefix_baseline,
                        use_position_baseline=use_position_baseline,
                        use_level_baseline=use_level_baseline,
                        use_accuracy_baseline=use_accuracy_baseline,
                        use_process_rl=use_process_rl
                    )

                    if use_veri_bonus:
                        if current_last_reward > 0:
                            veri_reward = veri_reward * 2
                        else:
                            veri_reward = veri_reward
                    else:
                        veri_reward = veri_reward

                    current_reward_dist += "V+" if veri_reward_ > 0 else "V-"

                    full_ids += current_tokenized
                    token_rewards += [veri_reward] * len(current_tokenized)

                    average_part_reward += sum([veri_reward] * len(current_tokenized)) / len(current_tokenized)

                    answer_masks += [1] * len(current_tokenized)
                    round2toN_answer_masks += [0] * len(current_tokenized)
                    part_idx_masks += [current_part_idx] * len(current_tokenized)
                    split_token_masks += [0] * len(current_tokenized)

                    veri_rewards.append(veri_reward)
                    veri_answers.append(veri_answer)
                    final_rewards_for_logging.append(veri_reward)

                elif last_split_token == str_retry_token_ids or last_split_token == str_retry_token_ids_other:
                    current_part_idx += 1

                    current_tokenized = [int(x) for x in subtext_ids_str.split("_") if x]
                    subtext = tokenizer.decode(current_tokenized)

                    retry_r_, _, cg_ = get_rewards([gold_solution], [gold_extracted_answer], [subtext], strict=False)
                    retry_r_ = retry_r_[0]
                    cg_ = cg_[0]

                    current_reward = self.compute_advantage(
                        reward=retry_r_,
                        level=level,
                        accuracy=accuracy,
                        current_part_idx=current_part_idx,
                        current_prefix=current_reward_dist,
                        use_prefix_baseline=use_prefix_baseline,
                        use_position_baseline=use_position_baseline,
                        use_level_baseline=use_level_baseline,
                        use_accuracy_baseline=use_accuracy_baseline,
                        use_process_rl=use_process_rl
                    )

                    if use_bonus:
                        if retry_r_ != current_last_reward:
                            current_reward = current_reward * 2
                        else:
                            current_reward = current_reward
                    else:
                        current_reward = current_reward

                    current_reward_dist += "R+" if retry_r_ > 0 else "R-"

                    full_ids += current_tokenized
                    token_rewards += [current_reward] * len(current_tokenized)
                    average_part_reward += sum([current_reward] * len(current_tokenized)) / len(current_tokenized)
                    answer_masks += [1] * len(current_tokenized)
                    round2toN_answer_masks += [1] * len(current_tokenized)
                    part_idx_masks += [current_part_idx] * len(current_tokenized)
                    split_token_masks += [0] * len(current_tokenized)

                    current_last_reward = retry_r_
                    final_rewards_for_logging.append(current_reward)
                    response_extracted_answers.append(cg_)
                    response_rewards += [current_reward]

                else:
                    raise ValueError("Last split token is not valid.")

            part_len = len(final_rewards_for_logging)
            if part_len not in group_reward_dict:
                group_reward_dict[part_len] = []
            else:
                group_reward_dict[part_len].append(final_rewards_for_logging)

            total_reward = sum(final_rewards_for_logging)
            average_reward = average_part_reward/part_len

            example["reward"] = current_last_reward

            if current_reward_dist != reward_dist:
                print_rank_0(f"sample idx: {idx} reward dist is inconsistent：{current_reward_dist} vs {reward_dist}, have been removed.")
                misaligned_idx_list.append(idx)
                continue

            example["average_reward"] = average_reward
            example["token_rewards"] = token_rewards
            example["answer_masks"] = answer_masks
            example["part_idx_masks"] = part_idx_masks
            example["round1_answer_masks"] = round1_answer_masks
            example["split_token_masks"] = split_token_masks
            example["round2toN_answer_masks"] = round2toN_answer_masks

            has_split = any(x for x in split_token_masks if x == 1)
            example["valid_splits"] = True if has_split else False
            example["response_rewards"] = response_rewards
            example["veri_rewards"] = veri_rewards
            example["veri_answers"] = veri_answers
            example["gold_extracted_answer"] = gold_extracted_answer
            example["response_extracted_answers"] = response_extracted_answers

            updated_dataset.append(example)

        self.group_reward_baseline = self.compute_reward_baseline(group_reward_dict)
        if fp_save_path:
            self.save_preprocess(fp_save_path, updated_dataset)
            print_rank_0(f"Preprocess data saved to {fp_save_path}")

        return updated_dataset, misaligned_idx_list
    
    def get_baseline(self, chain_length:int, part_idx:int):
        reward = self.group_reward_baseline[chain_length][part_idx]
        return reward


    @staticmethod
    def compute_reward_baseline(group_reward_dict:Dict):
        """
            to get reward of ith part as baseline from group_reward_dict 
        """
        group_reward_baseline = {}
        for part_len, rewards_list in group_reward_dict.items():
            rewards_tensor = torch.tensor(rewards_list, dtype=torch.float)
            rewards_mean = rewards_tensor.mean(dim=0)
            group_reward_baseline[part_len] = rewards_mean
        return group_reward_baseline


    @staticmethod
    def save_preprocess(fp_save:Union[str, Path], dataset:List[Dict]):
        with open(fp_save, "w") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)


@dataclass
class DataCollatorForOfflineRL:
    """Collate examples for PPO fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:

        data_collator_dict = self.offline_data_collator(instances, self.tokenizer)

        return data_collator_dict
    
    @staticmethod
    def offline_data_collator(instances: List[Dict], tokenizer) -> Dict[str, torch.Tensor]:

        input_ids_list = []
        labels_list = []
        token_rewards_list = []
        answer_masks_list = []
        part_idx_masks_list = []
        round1_answer_masks_list = []
        split_token_masks_list = []
        round2toN_answer_masks_list = []
        reward_list = []
        weights_list = []
        sft_mask_list = []

        for instance in instances:
            input_ids_list.append(torch.tensor(instance["input_ids"], dtype=torch.long))
            labels_list.append(torch.tensor(instance["labels"], dtype=torch.long))
            token_rewards_list.append(torch.tensor(instance["token_rewards"], dtype=torch.float))
            answer_masks_list.append(torch.tensor(instance["answer_masks"], dtype=torch.long))
            part_idx_masks_list.append(torch.tensor(instance["part_idx_masks"], dtype=torch.long))
            round1_answer_masks_list.append(torch.tensor(instance["round1_answer_masks"], dtype=torch.long))
            split_token_masks_list.append(torch.tensor(instance["split_token_masks"], dtype=torch.long))
            round2toN_answer_masks_list.append(torch.tensor(instance["round2toN_answer_masks"], dtype=torch.long))
            sft_mask_list.append(torch.tensor(instance["sft_mask"], dtype=torch.float))
            weights_list.append(torch.tensor(instance["weight"], dtype=torch.float))
            reward_list.append(torch.tensor(instance["reward"], dtype=torch.float))

        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
        labels = pad_sequence(labels_list, batch_first=True, padding_value=IGNORE_INDEX)
        attention_masks = input_ids.ne(tokenizer.pad_token_id).long()
        token_rewards = pad_sequence(token_rewards_list, batch_first=True, padding_value=0.0)
        answer_masks = pad_sequence(answer_masks_list, batch_first=True, padding_value=0)
        part_idx_masks = pad_sequence(part_idx_masks_list, batch_first=True, padding_value=0)
        round1_answer_masks = pad_sequence(round1_answer_masks_list, batch_first=True, padding_value=0)
        split_token_masks = pad_sequence(split_token_masks_list, batch_first=True, padding_value=0)
        round2toN_answer_masks = pad_sequence(round2toN_answer_masks_list, batch_first=True, padding_value=0)

        sft_mask = torch.stack(sft_mask_list, dim=0)
        weights = torch.stack(weights_list, dim=0)
        rewards = torch.stack(reward_list, dim=0)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "rewards": rewards,
            "attention_mask": attention_masks,
            "token_rewards": token_rewards,
            "weights": weights,
            "answer_masks": answer_masks,
            "part_idx_masks": part_idx_masks,
            "round1_answer_masks": round1_answer_masks,
            "split_token_masks": split_token_masks,
            "round2toN_answer_masks": round2toN_answer_masks,
            "sft_mask": sft_mask
        }