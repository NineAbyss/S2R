"""Util functions for the project."""

import json
import random

import torch
from tools.answer_extraction import strip_string, extract_answer, answer_corrected_match
import re

# For debug
def is_rank_0():
    """check if at rank 0 gpu"""
    print(torch.distributed.is_initialized())
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


def print_rank_0(message):
    """print message for multi-gpu case"""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


# Utils for loading and saving json and jsonl files
def jdump(data, output_path):
    """Dump a str or dictionary to a file in .json or .jsonl format."""
    if is_rank_0():
        with open(output_path, "w") as f:
            if output_path.endswith(".json"):
                json.dump(data, f, ensure_ascii=False, indent=4)
            elif output_path.endswith(".jsonl"):
                for entry in data:
                    json.dump(entry, f, ensure_ascii=False)
                    f.write("\n")
            else:
                raise ValueError("File must be a .json or .jsonl file")
        f.close()


def jload(data_path):
    """Load a .json or .jsonl file into a dictionary or list of dictionaries."""

    with open(data_path, "r") as f:
        if f.name.endswith(".json"):
            data = json.load(f)
        elif f.name.endswith(".jsonl"):
            lines = f.read().strip().split("\n")
            data = [json.loads(line) for line in lines]
        else:
            raise ValueError("File must be a .json or .jsonl file")

        f.close()
    return data



def is_main_process():
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            return True
        else:
            return False
    else:
        return True


def print_rank_0_color(message, end='\n', color='green') -> None:
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


def extract_and_process_boxed_answers(text):
    answers = []
    for piece in text.split('boxed{')[1:]:
        n = 0
        for i in range(len(piece)):
            if piece[i] == '{':
                n += 1
            elif piece[i] == '}':
                n -= 1
                if n < 0:
                    if i + 1 < len(piece) and piece[i + 1] == '%':
                        answers.append(piece[: i + 1])
                    else:
                        answers.append(piece[:i])
                    break
    
    if answers:
        answer = answers[-1]
    else:
        return ""
    

    # 用已有代码对答案进行处理
    answer = answer.strip().split("\n")[0]
    answer = answer.lstrip(":")
    answer = answer.rstrip(".")
    answer = answer.rstrip("/")
    answer = strip_string(answer)
    
    # 如果答案中有等号，取等号后面的部分
    if "=" in answer:
        answer = answer.split("=")[1].strip()
    
    # 处理\\frac{29}4这种情况，转换为\\frac{29}{4}
    answer = re.sub(r"\\frac{(\d+)}(\d+)", r"\\frac{\1}{\2}", answer)
    
    return answer


# from answer_extraction import 

def get_soft_answer_correction(gold_answer, output_answer):
    if gold_answer == output_answer:
        return True
    if answer_corrected_match(gold_answer, output_answer) or answer_corrected_match(output_answer, gold_answer):
        # print(f"Corrected answer match:\n gold: {gold_answer}, output: {output_answer}")
        return True
    
    return False


def get_rewards(gold_solution_list, gold_answer_list, responses, strict=False):
    if strict:
        extract_answer_func = extract_and_process_boxed_answers
    else:
        extract_answer_func = extract_answer
    rewards = []
    gold_answers = []
    generated_answers = []
    for solution, answer, response in zip(gold_solution_list, gold_answer_list, responses):

        gold_answer = extract_answer_func(solution)
        if "=" in gold_answer:
            gold_answer = gold_answer.split("=")[1].strip()
            
        generated_answer = extract_answer_func(response)
        if "=" in generated_answer:
            generated_answer = generated_answer.split("=")[1].strip()
        
        if get_soft_answer_correction(gold_answer, generated_answer) or get_soft_answer_correction(answer, generated_answer):
            rewards.append(1)
        else:
            rewards.append(-1)
        
        gold_answers.append(gold_answer)
        generated_answers.append(generated_answer)
    
    return rewards, gold_answers, generated_answers


def get_rewards_and_select_responses(gold_solution_list, responses, ref_answers):
    '''
    Select responses for multiple generated answers
    responses: List[List[str]]
    '''
    rewards = []
    gold_answers = []
    generated_answers = []
    selected_responses = []
    selected_response_idxs = []
    for solution, response_list, ref_answer in zip(gold_solution_list, responses, ref_answers):
        gold_answer = extract_answer_func(solution)
        if "=" in gold_answer:
            gold_answer = gold_answer.split("=")[1].strip()
        
        reward_list = []
        generated_answer_list = []
        for response in response_list:
            generated_answer = extract_answer_func(response)
            if "=" in generated_answer:
                generated_answer = generated_answer.split("=")[1].strip()
            generated_answer_list.append(generated_answer)
            
            if gold_answer == generated_answer:
                reward_list.append(1)
            else:
                reward_list.append(-1)

        selected_response = None
        generated_answer = None
        if 1 in reward_list:
            correct_index = reward_list.index(1)
            selected_response = response_list[correct_index]

            rewards.append(1)
            selected_responses.append(selected_response)
            generated_answers.append(gold_answer)
            selected_response_idxs.append(correct_index)
        else:
            for idx, (ans, res) in enumerate(zip(generated_answer_list, response_list)):
                if ans == ref_answer and ref_answer != "":
                    selected_response = res
                    generated_answer = ans
                    selected_response_idx = idx
                    break
            if selected_response is None:
                selected_idx = random.choice(range(len(response_list)))
                selected_response = response_list[selected_idx]
                generated_answer = generated_answer_list[selected_idx]
                selected_response_idx = selected_idx
                
            rewards.append(-1)
            selected_responses.append(selected_response)
            generated_answers.append(generated_answer)
            selected_response_idxs.append(selected_response_idx)
        
        
        gold_answers.append(gold_answer)
    
    assert len(rewards) == len(selected_responses) == len(gold_answers) \
            == len(generated_answers) == len(gold_solution_list) \
                == len(selected_response_idxs)
    
    return rewards, gold_answers, generated_answers, selected_responses, selected_response_idxs



class ExponentialMovingAverage:

    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.ema = None

    def update(self, num):
        prev_ema = num if self.ema is None else self.ema
        self.ema = self.alpha * prev_ema + (1.0 - self.alpha) * num
        return self.ema

    def get(self):
        return self.ema if self.ema is not None else 0.


def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output


import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator



def get_train_ds_config(offload,
                        dtype,
                        stage=2,
                        enable_hybrid_engine=False,
                        inference_tp_size=1,
                        release_inference_cache=False,
                        pin_parameters=True,
                        tp_gather_partition_size=8,
                        max_out_tokens=512,
                        enable_tensorboard=False,
                        enable_mixed_precision_lora=False,
                        tb_path="",
                        tb_name=""):

    device = "cpu" if offload else "none"
    if dtype == "fp16":
        data_type = "fp16"
        dtype_config = {"enabled": True, "loss_scale_window": 100}
    elif dtype == "bf16":
        data_type = "bfloat16"
        dtype_config = {"enabled": True}
    zero_opt_dict = {
        "stage": stage,
        "offload_param": {
            "device": device
        },
        "offload_optimizer": {
            "device": device
        },
        "stage3_param_persistence_threshold": 1e4,
        "stage3_max_live_parameters": 3e7,
        "stage3_prefetch_bucket_size": 3e7,
        "memory_efficient_linear": False
    }
    if enable_mixed_precision_lora:
        zero_opt_dict["zero_quantized_nontrainable_weights"] = True
        if dist.get_world_size() != get_accelerator().device_count():
            zero_opt_dict["zero_hpz_partition_size"] = get_accelerator(
            ).device_count()
    return {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        data_type: dtype_config,
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "hybrid_engine": {
            "enabled": enable_hybrid_engine,
            "max_out_tokens": max_out_tokens,
            "inference_tp_size": inference_tp_size,
            "release_inference_cache": release_inference_cache,
            "pin_parameters": pin_parameters,
            "tp_gather_partition_size": tp_gather_partition_size,
        },
        "tensorboard": {
            "enabled": enable_tensorboard,
            "output_path": f"{tb_path}/ds_tensorboard_logs/",
            "job_name": f"{tb_name}_tensorboard"
        }
    }

