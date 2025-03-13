import json
import sys, re
from pathlib import Path
import numpy as np
import argparse
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Union
from tools.answer_extraction import extract_answer, answer_corrected_match

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acc_min", default=0.1, type=float)
    parser.add_argument("--acc_max", default=0.7, type=float)
    parser.add_argument("--fp_in", type=str)
    parser.add_argument("--fp_save", type=str)
    parser.add_argument("--response_key", type=str, default="code")
    parser.add_argument("--use_data_balance", type=bool, default=True)
    args = parser.parse_args()
    return args


def load_json_list(fp: Union[Path, str], encoding='utf-8') -> List[dict]:
    """load a txt file where each row as a dict.

    :param fp: the file path of the event-list
    :param encoding: the encoding manner

    :return: a list of dicts
    """
    with open(fp, encoding=encoding) as f:
        json_lines = list(map(lambda x: json.loads(x), f.readlines()))
    return json_lines


def save_json_dict(
        dct, fname: Union[str, Path], encoding='utf-8',
        indent=2, **kwargs):
    with open(fname, 'w', encoding=encoding) as jsfile:
        json.dump(dct, jsfile, ensure_ascii=False, indent=indent, **kwargs)


def print_dist_list(dist_list: Dict[str, int]):
    """
    tool function: 用于打印 reward context 的分布情况

    :Args:

    :param dist_list: Dict[str, int] = {
        "R+V+": int,
        "R+V-": int,
        ...
    }
    """
    print(f"-"*10,"Reward Distribution","-"*10)
    dist_list = dict(sorted(dist_list.items(), key=lambda x: x[1], reverse=True))
    print(dist_list)
    a = [dist for dist in dist_list.keys() if dist_list[dist] > 0]

    length = sum([dist_list[key] for key in a])
    print(f"total: {length}")
    for key in a[:4]:
        print(f"{key}: {dist_list[key]/length:.2f}")
    
    # 打印 最长的 reward dist 的长度
    max_len = max([len(dist)//2 for dist in dist_list.keys()])
    print(f"max_len: {max_len}")
    return dist_list


def get_soft_answer_correction(gold_answer, output_answer):
    if "=" in gold_answer:
        gold_answer = gold_answer.split("=")[-1].strip()
    if "=" in output_answer:
        output_answer = output_answer.split("=")[-1].strip()
    
    if gold_answer == output_answer:
        return True
    if answer_corrected_match(gold_answer, output_answer) or answer_corrected_match(output_answer, gold_answer):
        # print(f"Corrected answer match:\n gold: {gold_answer}, output: {output_answer}")
        return True
    
    return False


def get_single_reward_dist(response:str, gold_answer:str, gold_extracted_answer:str=None) -> str:
    """
    tool function: extract reward context of a single response

    :Args:
    
    :param response: str,
    :param gold_answer: str,
    :param gold_extracted_answer: str

    :returns dist: str, reward context
    """
    if gold_extracted_answer is None:
        gold_extracted_answer = gold_answer

    dist = ""
    check_token1 = "Wait,"
    check_token2 = "Let me recheck my solution."
    retry_token = "Let me try again.\n\n"
    last_split_token = ""
    
    split_texts = re.split(f"({re.escape(check_token1)}|{re.escape(retry_token)}|{re.escape(check_token2)})", response)
    
    all_answers = []
    all_verifications = []
    
    initial_text = split_texts[0]
    initial_answer = extract_answer(initial_text)
    all_answers.append(initial_answer)
    
    final_answer = initial_answer
    final_answer_is_correct = get_soft_answer_correction(final_answer, gold_answer) or\
        get_soft_answer_correction(final_answer, gold_extracted_answer)
    
    reward = 1 if final_answer_is_correct else -1
    dist += "R-" if reward == -1 else "R+"

    for i, subtext in enumerate(split_texts[1:]):
        
        if not subtext.strip():
            continue
        
        # split token
        if subtext in [check_token1, check_token2] or subtext == retry_token:
            last_split_token = subtext
            continue
        
        # check
        elif last_split_token in [check_token1, check_token2]:
            
            if "is incorrect" in subtext or "cannot verify"  in subtext: 
                veri_answer = "incorrect"
            elif "is correct" in subtext or "verified" in subtext or "is indeed" in subtext or\
                 "verified to be correct" in subtext or "verified as correct" in subtext:
                veri_answer = "correct"
            else:
                if i != len(split_texts[1:]) - 2:
                    veri_answer = "incorrect"
                else:
                    veri_answer = "correct"
            
            if final_answer_is_correct and veri_answer == "correct" or not final_answer_is_correct and veri_answer == "incorrect":
                reward = 1
            else:
                reward = -1
            dist += "V-" if reward == -1 else "V+"
            all_verifications.append(veri_answer)

        # retry
        elif last_split_token == retry_token:
            
            final_answer = extract_answer(subtext)
            final_answer_is_correct = get_soft_answer_correction(final_answer, gold_answer)

            reward = 1 if final_answer_is_correct else -1
            dist += "R-" if reward == -1 else "R+"
            all_answers.append(final_answer)
            
    return dist


def get_all_reward_dist(
    all_lines: List[Dict[str, Any]], 
    response_key:str="code", 
    gold_answer_key:str="gt"
)-> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    tool function: add reward context to all samples

    :Args:

    :param all_lines: List[Dict[str, Any]] = [
            {
                "code": list = [str, str, ...],
                "gt": str,
                ...
            },
            ...
        ]

    :param response_key: response key of the problem,
    :param gold_answer_key: golden answer of the problem
    """

    dist_list = defaultdict(int)
    for line in all_lines:
        reward_list = []
        try:
            response_list = line[response_key] if isinstance(line[response_key], list) else [line[response_key]]
            gold_answer = line[gold_answer_key]
            gold_extracted_answer = line[gold_answer_key]
        except:
            continue
        
        for response in response_list:
            reward_dist = get_single_reward_dist(response, gold_answer, gold_extracted_answer)
            reward_list.append(reward_dist)
            dist_list[reward_dist] += 1
        
        line["reward_list"] = reward_list
    # print_dist_list(dist_list)
    return all_lines, dist_list


def check_alter_reward(reward_str: str) -> bool:
    """
    tool function: Check if the reward context is valid("R*V*R*V*...")
    """
    if not reward_str:
        return False
    if len(reward_str) % 2 != 0:
        return False

    tokens = [reward_str[i:i+2] for i in range(0, len(reward_str), 2)]

    if len(tokens) % 2 != 0:
        return False
    
    for i, token in enumerate(tokens):
        expected_char = "R" if i % 2 == 0 else "V"
        if token[0] != expected_char:
            return False
    return True


def remain_fixed_rv(
    reward_dist:str="",
    answer:str="",
    gold_answer:str="", 
    gold_extracted_answer:str=""
)-> Tuple[str, str]:
    """
    tool function: transform reward context to fixed reward_dist;

    for example:
    Inputs are:
        reward_dist = "R+V+"
        reward dist of current answer = "R+V+R+V+R+V+"
    Outputs are:
        then reward dist of truncated answer = "R+V+"
    """

    current_dist = ""
    remain_answer = ""
    check_token = "Wait,"
    check_token2 = "Let me recheck my solution."
    retry_token = "Let me try again.\n\n"
    last_split_token = ""
    
    split_texts = re.split(f"({re.escape(check_token)}|{re.escape(retry_token)}|{re.escape(check_token2)})", answer)

    all_answers = []
    all_verifications = []
    
    initial_text = split_texts[0]
    initial_answer = extract_answer(initial_text)
    all_answers.append(initial_answer)
    
    final_answer = initial_answer
    final_answer_is_correct = get_soft_answer_correction(final_answer, gold_answer) or\
          get_soft_answer_correction(final_answer, gold_extracted_answer)
    
    reward = 1 if final_answer_is_correct else -1
    current_dist += "R-" if reward == -1 else "R+"
    remain_answer += initial_text

    for i, subtext in enumerate(split_texts[1:]):
        
        if current_dist == reward_dist:
            break

        if not subtext:
            continue
        
        # split token
        if subtext in [check_token2, check_token] or subtext == retry_token:
            last_split_token = subtext
            continue
        
        # check
        elif last_split_token in [check_token, check_token2]:
            remain_answer += last_split_token + subtext
            if "is incorrect" in subtext or "cannot verify"  in subtext: 
                veri_answer = "incorrect"
            elif "is correct" in subtext or "verified" in subtext or "is indeed" in subtext or\
                 "verified to be correct" in subtext or "verified as correct" in subtext:
                veri_answer = "correct"
            else:
                if i != len(split_texts[1:]) - 2:
                    veri_answer = "incorrect"
                else:
                    veri_answer = "correct"
            
            if final_answer_is_correct and veri_answer == "correct" or not final_answer_is_correct and veri_answer == "incorrect":
                reward = 1
            else:
                reward = -1
            current_dist += "V-" if reward == -1 else "V+"
            all_verifications.append(veri_answer)

        # retry
        elif last_split_token == retry_token:
            remain_answer += " " + retry_token + subtext
            
            final_answer = extract_answer(subtext)
            final_answer_is_correct = get_soft_answer_correction(final_answer, gold_answer)

            reward = 1 if final_answer_is_correct else -1
            current_dist += "R-" if reward == -1 else "R+"
            all_answers.append(final_answer)
    
    return remain_answer, current_dist


def process_special_answer(reward_str:str, answer:str="", gold_answer:str="", gold_extracted_answer:str="") -> Tuple[str, str]:
    """
    tool function: to process special answer, truncate the answer to the last R+V* context or first R+V+ context
    for example:
        "R+V+R+V+" -> "R+V+"
        "R+V+R-V+R+V-R+V-R-V+" -> "R+V+R-V+R+V-"
    """
    tokens = [reward_str[i:i+2] for i in range(0, len(reward_str), 2)]
    curr = 0
    last_token = ""
    for i, token in enumerate(tokens):
        if token == "R+":
            curr = i
        elif token == "V+":
            if last_token == "R+":
                break
        last_token = token

    reward_dist = "".join(tokens[:curr+2])
    remain_answer, current_dist = remain_fixed_rv(reward_dist, answer, gold_answer, gold_extracted_answer)
    return remain_answer, current_dist


def stage1_filter_problem(all_lines:List[dict], acc_min:float=0.1, acc_max:float=0.7):
    """
    offline data filter stage1: filter the data by accuracy

    :Params:
    :param all_lines: List[dict] = [
            {
                "score":list = [True, False, ......],
                ...
            },
            ...
        ]
    all_lines[0].keys() = ['idx', 'question', 'gt_cot', 'gt', 'level', 'solution', 'answer', 'code', 'pred', 'report', 'score']
    """

    acc_stat = defaultdict(int)
    full_correct = 0
    full_wrong = 0
    filtered_lines = []
    for line in all_lines:
        scores = line.get("score", [])
        if not scores:
            continue
        acc = sum(scores) / len(scores)
        if acc_min <= acc <= acc_max:
            filtered_lines.append(line)
        
        if acc == 0:
            full_wrong += 1
        elif acc == 1:
            full_correct += 1
        acc_stat[acc] += 1
        line["accuracy"] = acc
    
    acc_stat = sorted(acc_stat.items(), key=lambda x:x[0])
    print(f"Filter {len(all_lines)} to {len(filtered_lines)}")
    print(f"--"*20)
    print(f"Accuracy distribution: {acc_stat}")
    return filtered_lines


def stage2_rejection_sampling(dataset: List[Dict], use_data_balance:bool=True) -> List[Dict]:
    """
    tool function: rejection sampling to enhance quality of the dataset

    :param dataset: List[Dict]
    :param use
    """

    new_lines = []
    dist_list = defaultdict(int)
    general_keys = ['gt_cot', 'gt', 'level', 'solution', 'accuracy']

    for line in dataset:
        reward_list = line["reward_list"]
        negative_samples = []
        pos_count = 0
        gold_answer = line["gt"]

        for idx, old_reward_dist in enumerate(reward_list):
            pos = True
            answer = line["code"][idx]

            # Trucate the answer to the last R+V* context or first R+V+ context
            if "R+" in old_reward_dist:
                answer, reward_dist = process_special_answer(old_reward_dist, answer, gold_answer, gold_answer)
                line["reward_list"][idx] = reward_dist

            else:
                pos = False
                reward_dist = old_reward_dist

            if check_alter_reward(reward_dist):
                # filter by the length of trajectory
                if len(reward_dist) > 20:
                    continue

                if not pos:
                    negative_samples.append(
                        {
                            "problem": line["question"],
                            "raw_response": line["code"][idx],
                            "answer" : answer,
                            "reward_dist": reward_dist,
                            "old_reward_dist": old_reward_dist,
                            **{key: line[key] for key in general_keys},
                        }
                    )

                else:
                    pos_count += 1
                    new_lines.append(
                        {
                            "problem": line["question"],
                            "raw_response": line["code"][idx],
                            "answer" : answer,
                            "reward_dist": reward_dist,
                            "old_reward_dist": old_reward_dist,
                            **{key: line[key] for key in general_keys},
                        }
                    )
                    dist_list[reward_dist] += 1
            
        # remain negative samples to balance the dataset
        if len(negative_samples) >= pos_count:
            if use_data_balance:
                negatives = np.random.choice(negative_samples, pos_count, replace=False)
            else:
                negatives = negative_samples
            new_lines.extend(negatives)
            for sample in negatives:
                dist_list[sample["reward_dist"]] += 1
                
        else:
            new_lines.extend(negative_samples)
            for sample in negative_samples:
                dist_list[sample["reward_dist"]] += 1

    print_dist_list(dist_list)
    return new_lines


def process_trainset_format(df:list):
    """
    tool function: process the dataset to the format of training set
    """
    prompt_template = (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>user\n{input}<|im_end|>\n"
    )
    anwer_template = (
        "<|im_start|>assistant\n"
        "{output}<|im_end|>\n"
    )

    for line in df:
        line["prompt"] = prompt_template.format(input=line["problem"])
        line["answer"] = anwer_template.format(output=line["answer"])
        line["gold_extracted_answer"] = line["gt"]
    return df


def main_offline_trainset_process(
    fp_in:str,
    fp_save:str,
    response_key:str="code",
    acc_min:float=0.1,
    acc_max:float=0.7,
    use_data_balance:bool=True
):
    all_lines = load_json_list(fp_in)
    df_stage1 = stage1_filter_problem(all_lines, acc_min, acc_max)
    df_stage1, _ = get_all_reward_dist(df_stage1, response_key=response_key)
    df_stage2 = stage2_rejection_sampling(df_stage1, use_data_balance)
    trainset = process_trainset_format(df_stage2)

    save_json_dict(trainset, fp_save)
    print(f"{fp_in} has been processed and saved to {fp_save}")


if __name__ == "__main__":
    args = parse_args()
    main_offline_trainset_process(args.fp_in, args.fp_save, args.response_key, args.acc_min, args.acc_max, args.use_data_balance)


    