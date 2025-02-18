import gc
import math
import os
import json
import textwrap
import sys
import time
import random
import pdb
import re
import itertools
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union, Literal
from copy import deepcopy
import deepspeed
import copy
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from contextlib import contextmanager
import torch.nn.functional as F
 
 
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from transformers import (
    PreTrainedModel,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer import DEFAULT_CALLBACKS, DEFAULT_PROGRESS_CALLBACK
from transformers.trainer_callback import CallbackHandler, PrinterCallback, TrainerState, TrainerControl, ExportableState
from transformers.trainer_utils import has_length, HPSearchBackend
from transformers.utils import is_sagemaker_mp_enabled, find_labels, can_return_loss
from deepspeed.runtime.zero.partition_parameters import GatheredParameters
from transformers.integrations import (
    get_reporting_integration_callbacks,
    hp_params,
)

from arguments import RLOOConfig
from .rloo_utils import ExpMinDataset, padding_any, pad  
from utils import print_object_on_main_process, print_rank_0, get_rewards, ExponentialMovingAverage, to_device, get_rewards_and_select_responses, extract_and_process_boxed_answers
from .base import BaseTrainer
from accelerate.scheduler import AcceleratedScheduler

 
 

TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
FSDP_MODEL_NAME = "pytorch_model_fsdp"

if is_wandb_available():
    import wandb

INVALID_LOGPROB = 1.0


def remove_hooks(model) -> None:
    """Removes the optimizer hooks from a DeepSpeed ZeRO-3 model."""
    if model.optimizer is not None and hasattr(model.optimizer, "parameter_offload"):
        optimizer_offload = model.optimizer.parameter_offload
    elif model.optimizer is not None:
        optimizer_offload = model.optimizer

    for param in iter_params(optimizer_offload.module, recurse=True):
        param.ds_active_sub_modules.clear()

    for hook in optimizer_offload.forward_hooks:
        hook.remove()
    for hook in optimizer_offload.backward_hooks:
        hook.remove()

    optimizer_offload.forward_hooks = []
    optimizer_offload.backward_hooks = []

def get_all_parameters(sub_module, recurse=False):
    return itertools.chain(sub_module.named_parameters(recurse=recurse), sub_module.ds_external_parameters())

def iter_params(module, recurse=False):
    return [param for _, param in get_all_parameters(module, recurse)]


def add_hooks(model) -> None:
    """Adds the optimizer hooks from a DeepSpeed ZeRO-3 model."""
    if model.optimizer is not None and hasattr(model.optimizer, "parameter_offload"):
        optimizer_offload = model.optimizer.parameter_offload
    elif model.optimizer is not None:
        optimizer_offload = model.optimizer
    optimizer_offload._register_hooks_recursively(optimizer_offload.module)

@contextmanager
def unwrap_model_for_generation(
    model, accelerator, is_peft_model: bool = False
) :
    """Context manager to unwrap a model for generation.
    For ZeRO-3 models, we gather the weights once to speed up generation.
    """
    unwrapped_model = accelerator.unwrap_model(model)
    if is_peft_model:
        unwrapped_model.pretrained_model.disable_adapter()
    if accelerator.state.deepspeed_plugin is not None and accelerator.state.deepspeed_plugin.zero_stage == 3:

        with deepspeed.zero.GatheredParameters(model.parameters()):
            remove_hooks(model)
            yield model
             
            add_hooks(model)
    else:
        yield unwrapped_model


def exact_div(a, b, custom_error_message=""):
    q = a // b
    if a != q * b:
        raise ValueError(f"{custom_error_message}, inexact division: {a} / {b} = {a / b}")
    return q

def compute_lm_loglikeli(logits, labels, shift=True):
    batch_size, seq_length, vocab_size = logits.shape
        
    if shift:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
    else:
        shift_logits = logits
        shift_labels = labels
    
     
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    
     
    shift_labels = shift_labels.to(shift_logits.device)
    neg_logprobs = loss_fct(shift_logits, shift_labels).reshape(batch_size, -1)  
    ignore_mask = shift_labels != -100
    
     
     
    return -1* neg_logprobs, ignore_mask


def compute_lm_loglikeli_2(logits, labels, shift=True):
     
    batch_size, seq_length, vocab_size = logits.shape
        
    if shift:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
    else:
        shift_logits = logits
        shift_labels = labels
    
    shift_logits = torch.clamp(shift_logits, min=-30, max=30)   
    

    with torch.cuda.amp.autocast(enabled=False):  
        shift_logits_fp32 = shift_logits.float()  
        log_probs = torch.nn.functional.log_softmax(shift_logits_fp32, dim=-1)
        log_probs = log_probs.to(logits.dtype)    
    shift_labels = shift_labels.to(log_probs.device)
    ignore_mask = shift_labels != -100   
    selected_log_probs = log_probs.gather(-1, (shift_labels*ignore_mask).unsqueeze(-1)).squeeze(-1)
    
    neg_logprobs = -selected_log_probs   
    
     
    neg_logprobs = neg_logprobs.reshape(batch_size, -1)   
     
    
    return neg_logprobs, ignore_mask.long()


def get_veri_answer(subtext, is_last=False):
    pattern = re.compile(r"Therefore, the answer is (.+?)\.")
    result = re.findall(pattern, subtext)
    if len(result) > 0:
        for res in result[::-1]:
            if res in ["correct", "incorrect"]:
                return res
        
    if "is incorrect" in subtext or "cannot verify"  in subtext: 
        return "incorrect"
    elif "is correct" in subtext or "is indeed" in subtext or "verified to be correct" in subtext or "verified as correct" in subtext:
        return "correct"
    else:
        if is_last:
            return "correct"
        else:
            return "incorrect"


class RLOOTrainer(Trainer):

    def __init__(
        self,
        config: RLOOConfig,
        processing_class,
        policy: nn.Module,
        ref_policy: nn.Module,
        train_dataset: Dataset,
        data_collator=None,
        eval_dataset=None,
        sft_data_module=None,
         
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        callbacks: Optional[List[TrainerCallback]] = None,
    ) -> None:
        super().__init__(
            model=policy,
            args=config,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=processing_class
        )
        
        self.args = config
        args = config
        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        if self._is_create_ref_model():
            self.ref_model = ref_policy
            for param in self.ref_model.parameters():
                param.requires_grad = False

            if self.is_deepspeed_enabled:
                self.ref_model, self.ref_model_engine = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
         
         
        self.train_dataset_len = len(train_dataset)
         
             
         
         
        if args.total_episodes is None:   
            args.total_episodes = int(args.num_train_epochs * self.train_dataset_len)
         
         
         
        args.local_batch_size = (
            args.per_device_train_batch_size * args.gradient_accumulation_steps * args.num_mini_batches
        )
        args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
        args.batch_size = int(args.local_batch_size * args.world_size)
        args.mini_batch_size = exact_div(
            args.batch_size, args.num_mini_batches, "`batch_size` must be a multiple of `num_mini_batches`"
        )
        self.args.local_mini_batch_size = exact_div(
            args.local_batch_size, args.num_mini_batches, "`local_batch_size` must be a multiple of `num_mini_batches`"
        )
        args.num_total_batches = math.ceil(
            args.total_episodes / args.batch_size
        )   
         
         
         
         
        if args.num_sample_generations > 0:
            self.sample_generations_freq = max(1, args.num_total_batches // args.num_sample_generations)
        
         
        self.local_dataloader_batch_size = exact_div(
            args.local_batch_size, 1, "`local_batch_size` must be a multiple of rloo_k"
        )   


        if args.stop_token and args.stop_token == "eos":
            args.stop_token_id = self.tokenizer.eos_token_id




     
    def initialize_baseline(self, use_level=False, use_prefix=False):
        if use_prefix:
            self.global_prefix_sums = defaultdict(float)
            self.global_prefix_counts = defaultdict(float)
            
            if "restart" in self.args.output_dir:
                prefix_data_path = os.path.join(self.args.model_path.split("checkpoint-")[0], "prefix_baseline.json")
                print_rank_0(f"Loading prefix data from {prefix_data_path}")
                with open(prefix_data_path, "r") as f:
                    prefix_data = json.load(f)
                for key in prefix_data["global_prefix_sums"].keys():
                    if key not in self.global_prefix_sums:
                        self.global_prefix_sums[key] = prefix_data["global_prefix_sums"][key]
                        self.global_prefix_counts[key] = prefix_data["global_prefix_counts"][key]
                
                self.prefix2idx = {key: idx for idx, key in enumerate(prefix_data["global_prefix_sums"].keys())}
                self.idx2prefix = {idx: key for idx, key in enumerate(prefix_data["global_prefix_sums"].keys())}
                
                return
            

            key_list = ['', 'R1', 'R1V1', 'R-1', 'R-1V1', 'R-1V1R-1', 'R-1V-1', 'R-1V1R-1V1', 'R-1V1R-1V1R-1', 'R1V-1', 'R-1V1R-1V1R-1V1', 'R-1V1R-1V-1', 'R1V-1R1', 'R-1V1R1', 'R-1V1R-1V1R-1V1R-1', 'R1V-1R1V1', 'R-1V1R1V1', 'R-1V1R-1V1R-1V1R-1V1', 'R-1V1R-1V1R-1V-1', 'R1V-1R-1', 'R-1V-1R-1', 'R-1V1R-1V1R-1V1R-1V1R-1', 'R-1V1R-1V1R-1V1R-1V-1', 'R1V-1R-1V1', 'R-1V1R1V-1', 'R1V1R1', 'R-1V1R-1V1R1', 'R1V-1R1V-1', 'R-1V1R-1V1R-1V1R-1V1R-1V1', 'R1V1R1V1', 'R-1V-1R-1V1', 'R1V-1R1V-1R1', 'R-1V-1R-1V-1', 'R1V-1R-1V1R-1', 'R-1V-1R-1V1R-1', 'R-1V1R1V-1R-1', 'R1V-1R-1V1R1', 'R-1V1R1V-1R1', 'R-1V1R-1V-1R-1', 'R1V-1R-1V-1', 'R-1V1R-1V1R-1V1R-1V1R-1V1R-1', 'R-1V1R-1V1R-1V1R-1V1R-1V-1', 'R-1V1R-1V1R1V-1', 'R-1V1R-1V1R1V1', 'R1V-1R-1V1R-1V1', 'R-1V1R-1V1R-1V1R1', 'R-1V1R1V-1R-1V1', 'R-1V1V1', 'R-1V-1R-1V1R-1V1', 'R1V-1R1V-1R1V1', 'R-1V-1R1', 'R1V-1R1V-1R-1', 'R1V-1R-1V1R1V1', 'R1V-1R-1V1R-1V1R-1', 'R-1V1R-1V1R-1V1R-1V1R-1V1R-1V1', 'R-1V1R-1V-1R-1V1', 'R-1V1R1V-1R1V1', 'R-1V1R-1V1R1V-1R-1', 'R-1V1R1V-1R-1V1R-1', 'R1V-1R1V-1R1V-1', 'R1V1R-1', 'R-1V-1R-1V1R-1V1R-1', 'R-1V1V1R-1', 'R-1V1R-1V1R-1V1R1V-1', 'R-1V1R1V-1R1V-1', 'R-1V1R-1V-1R-1V-1', 'R1V-1R-1V1R-1V1R-1V1', 'R-1V-1R-1V1R-1V-1', 'R1V-1R1V-1R-1V1', 'R-1V1R-1V1R-1V-1R-1', 'R-1V1R-1V1R-1V1R-1V1R-1V1R-1V-1', 'R1V-1R-1V1R1V-1', 'R-1V1R-1V1R-1V1R-1V1R-1V1R-1V1R-1', 'R-1V1R-1V-1R-1V1R-1', 'R1V-1R1V-1R1V-1R1', 'R-1V1R-1V1R1V-1R-1V1', 'R-1V-1R1V1', 'R-1V1R1V-1R-1V1R-1V1', 'R-1V1R-1V1V1', 'R-1R-1', 'R-1V1R1V-1R-1V-1', 'R-1V1R-1V1R-1V1R1V1', 'R-1V1R-1V1R1V-1R1', 'R1V1R-1V1', 'R-1V1V1R-1V1', 'R1V-1R-1V1R-1V-1', 'R-1V1V-1', 'R-1V1R-1V1R-1V1R1V-1R-1', 'R-1V1R-1V1R1V-1R-1V1R-1', 'R-1V-1R-1V-1R-1', 'R-1V1R-1V1R-1V1R-1V1R-1V1R-1V1R-1V1', 'R1V-1R-1V1R-1V1R1', 'R1V1R1V-1', 'R-1V-1R-1V1R-1V1R-1V1', 'R1V-1R-1V1R-1V1R-1V1R-1', 'R-1R-1V1', 'R-1V1R1V-1R1V-1R1', 'R-1V1R1V-1R-1V1R1', 'R-1V1R1V-1R-1V1R-1V1R-1', 'R-1V-1R1V-1', 'R-1V-1R-1V1R-1V1R-1V-1', 'R-1V1V1R-1V1R-1', 'R-1V1R-1V1R-1V1R-1V1R1', 'R1V-1R-1V1R1V-1R1', 'R-1V1R-1V1V1R-1', 'R1V1R1V-1R1', 'R-1V1R1V-1R1V-1R-1', 'R1V-1R1V-1R1V-1R1V1', 'R1V1R-1V1R1', 'R-1V1R-1V1R-1V-1R-1V-1', 'R-1V1R-1V1R-1V-1R-1V1', 'R1V-1R1V-1R-1V1R1', 'R-1V1R-1V1R-1V1R1V-1R-1V1', 'R1V-1R1V-1R-1V1R-1', 'R1V-1R1V-1R-1V-1', 'R-1V1R-1V-1R-1V1R-1V-1', 'R-1V1R-1V1V-1', 'R-1V1R-1V1R1V-1R1V-1', 'R-1V-1R-1V-1R-1V-1', 'R-1V1R-1V-1R-1V1R-1V1', 'R-1R-1V1R-1', 'R1V-1R-1V1R1V-1R-1', 'R-1V-1R-1V1R1', 'R-1V1R-1V1R1V-1R-1V-1', 'R1V1R-1V-1', 'R-1V1V1R-1V-1', 'R-1V-1V1', 'R1V-1R1V1R1', 'R1V1R1V-1R1V1', 'R-1V1R-1V1R-1V1R-1V1R-1V1R-1V1R-1V1R-1', 'R1V1R-1V1R-1', 'R-1V1R-1V1R1V-1R-1V1R-1V1', 'R-1V-1R-1V1R-1V1R-1V1R-1', 'R1V-1R1V-1R1V-1R1V-1', 'R1V-1R-1V1R1V-1R1V-1', 'R1V-1R-1V1R-1V1R1V-1', 'R1V-1R1V-1R1V-1R-1', 'R-1V-1R1V-1R-1', 'R1V-1R-1V1R-1V1R1V1', 'R-1V1R-1V1R-1V-1R-1V1R-1', 'R-1V1R-1V1R-1V1V1', 'R-1V-1V1R-1', 'R-1V1R-1V1R-1V1R1V-1R-1V1R-1', 'R-1V1R1V-1R1V-1R-1V1', 'R-1V1R1V-1R-1V1R-1V-1', 'R-1V1V1R-1V1R-1V1', 'R-1V1R1V-1R-1V1R-1V1R-1V-1', 'R-1V1V1V1', 'R-1V1R-1V1R1V-1R-1V1R-1V-1', 'R1V-1R-1V1R-1V1R-1V1R-1V1', 'R-1V1R-1V1R1V-1R1V1', 'R1V-1R-1V1R-1V1R-1V-1', 'R1V1R1V1R1', 'R-1V1R-1V1R-1V1R-1V1R-1V1R-1V1R-1V-1', 'R-1V1R-1V1R-1V1R1V-1R1', 'R1V1R-1V1R1V1', 'R-1V1R1V-1R-1V1R-1V1R-1V1', 'R1V-1R-1V1R-1V1R-1V1R1', 'R-1V1R-1V1R1V-1R-1V1R-1V1R-1', 'R-1R-1V1R-1V1', 'R1V-1R-1V1R1V-1R-1V1', 'R-1R-1V-1', 'R-1V1R-1V1R-1V1R-1V1R1V1', 'R-1V1R1V-1R1V-1R1V1']

            self.prefix2idx = {key: idx for idx, key in enumerate(key_list)}
            self.idx2prefix = {idx: key for idx, key in enumerate(key_list)}
            
            for key in key_list:
                self.global_prefix_sums[key] = 0
                self.global_prefix_counts[key] = 0
            
        elif use_level:
            self.global_position_sums = defaultdict(lambda: defaultdict(float))   
            self.global_position_counts = defaultdict(lambda: defaultdict(float))   
        else:
            self.global_position_sums = defaultdict(float)   
            self.global_position_counts = defaultdict(int)   

    def dump_baseline(self, use_level=False, use_prefix=False):
        if self.state.is_world_process_zero:
            if use_prefix:
                with open(os.path.join(self.args.output_dir,"prefix_baseline.json"), "w") as f:
                    json.dump({"global_prefix_sums": self.global_prefix_sums, "global_prefix_counts": self.global_prefix_counts}, f)
            elif use_level:
                with open(os.path.join(self.args.output_dir,"level_baseline.json"), "w") as f:
                    json.dump({"global_position_sums": self.global_position_sums, "global_position_counts": self.global_position_counts}, f)
            else:
                with open(os.path.join(self.args.output_dir,"baseline.json"), "w") as f:
                    json.dump({"global_position_sums": self.global_position_sums, "global_position_counts": self.global_position_counts}, f)
                
                
    def _prepare_deepspeed(self, model: PreTrainedModel):
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)
        del config_kwargs['optimizer']
        del config_kwargs['scheduler']

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size" : 0.9 * hidden_size * hidden_size
                        }
                    )
        
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        engine, engine.optimizer, engine.training_dataloader, engine.lr_scheduler = deepspeed.initialize(model=model, config=config_kwargs)
        model = engine.module
        model.eval()
        return model, engine

    

    def _is_create_ref_model(self) -> bool:
        return True



    def synchronize_dict(self, data_dict, old_data_dict):
        """
        Synchronize a dictionary across all processes using the Accelerator.
        """
         
        all_keys = list(data_dict.keys())
        all_keys_length = torch.tensor([len(all_keys)], dtype=torch.int64).to(self.accelerator.device)
        max_length = self.accelerator.gather(all_keys_length).max().item()

         
        padded_keys = all_keys + [None] * (max_length - len(all_keys))
         
        all_keys_tensor = torch.tensor([self.prefix2idx[key_str] if key_str is not None else -1 for key_str in padded_keys ]).to(self.accelerator.device)
         
        gathered_keys = self.accelerator.gather(all_keys_tensor).cpu().numpy()

         
        unique_keys = set()
        for key_idx in gathered_keys:
            if key_idx != -1:   
                unique_keys.add(key_idx)

         
        unique_keys = sorted(unique_keys)

        to_gather_tensor = torch.zeros(max(unique_keys)+1, dtype=torch.float32).to(self.accelerator.device)
        for idx in unique_keys:
            to_gather_tensor[idx] = data_dict.get(self.idx2prefix[idx], 0.0)

        gathered_tensor = self.accelerator.reduce(to_gather_tensor, "sum").detach().cpu()
        
        for idx in unique_keys:
            old_data_dict[self.idx2prefix[idx]] += gathered_tensor[idx].item()

        return old_data_dict



    
    def generate_experience_incot_splitmask(self, model, data_dict, all_rewards, all_sample_num, stage="stage1", use_bonus=False, use_veri_bonus=False, use_baseline=False, use_level_baseline=False, use_prefix_baseline=False, rloo_n=1):
        '''
        raw_prompt_list: list of string (input problems)
        '''
        
        
        self.round_1_instruction_template = "Please reason step by step, and put your final answer within \\boxed{}."

        model.eval()  
        self.tokenizer.padding_side = "left"
        round1_input_ids_list = []
        for line in data_dict["problem"]:
            seq = [{"role": "system", "content": self.round_1_instruction_template},
                       {"role": "user", "content": line}]

            tokenized_inputs = self.tokenizer.apply_chat_template(seq, return_tensors="pt", add_generation_prompt=True, tokenize=True)[0]
            round1_input_ids_list.append(tokenized_inputs)

        
        round1_input_ids = padding_any(round1_input_ids_list, padding_strategy="longest", max_length=self.tokenizer.model_max_length, padding_side="left", pad_token=self.tokenizer.pad_token_id)
                            
        round1_input_ids = round1_input_ids.to(model.device)
        round1_attention_masks = round1_input_ids != self.tokenizer.pad_token_id  
        
        with torch.no_grad():
            input_length = round1_input_ids.size(1)
            if self.args.use_tf32_forward:
                with torch.autocast(dtype=torch.float32, device_type="cuda"):
                    round1_responses_ = model.generate(round1_input_ids, attention_mask=round1_attention_masks,max_length=self.tokenizer.model_max_length, max_new_tokens=None,
                                              num_return_sequences=rloo_n, temperature=0.7, top_p=1.0, do_sample=True, synced_gpus=True,
                                              pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
                             
            else:
                round1_responses_ = model.generate(round1_input_ids, attention_mask=round1_attention_masks,max_length=self.tokenizer.model_max_length, max_new_tokens=None,
                                              num_return_sequences=rloo_n, temperature=0.7, top_p=1.0, do_sample=True, synced_gpus=True,
                                              pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
            
            no_eos_mask_round1 = ((round1_responses_[:,-1] != self.tokenizer.eos_token_id) & (round1_responses_[:,-1] != self.tokenizer.pad_token_id)).detach().cpu()

         

        round1_decoded_responses = self.tokenizer.batch_decode(round1_responses_[:, input_length:], skip_special_tokens=True) 
                     
        round1_responses_ = round1_responses_.detach().cpu()
        round1_input_ids = round1_input_ids.detach().cpu()
        

        round1_responses = torch.full((round1_responses_.size(0), self.tokenizer.model_max_length), self.tokenizer.pad_token_id)
        round1_responses[:, :round1_responses_.size(1)] = round1_responses_
        

        torch.cuda.empty_cache()
        
        valid_splits = []
        last_rewards = []
        round1_rewards = []
        batch_split_texts = []
        batch_generated_answers = []
        
        all_split_rewards = []
        all_veri_answers = []
        all_veri_rewards = []
        all_final_rewards = []
        all_levels = []
         
        
        sft_data_flag = []
        

        token_rewards = torch.zeros_like(round1_responses, dtype=torch.float32)
        answer_masks = torch.zeros_like(round1_responses)
         
        split_token_masks = torch.zeros_like(round1_responses)
        
         
        part_idx_masks = torch.zeros_like(round1_responses)
        
         
        prefix_sums = defaultdict(float)
        prefix_counts = defaultdict(int)
        for idx, (response_text, response_token_ids) in enumerate(zip(round1_decoded_responses, round1_responses)):
            
            response_token_ids = response_token_ids.tolist()
            sft_data_flag.append(data_dict["is_sft"][idx//rloo_n])
            all_levels.append(data_dict["level"][idx//rloo_n])
            
            split_rewards = []
            veri_rewards = []
            veri_answers = []
            final_rewards = []
            current_answer_prefix = ""

            generated_answers = []

            str_response_token_ids = "_".join([str(token) for token in response_token_ids[input_length:]])
            str_response_token_ids = str_response_token_ids.split(str(self.tokenizer.eos_token_id))[0].strip("_")
             
             
            str_check_token_ids = "_".join([str(token) for token in self.tokenizer("Wait,").input_ids])
            str_retry_token_ids = "_".join([str(token) for token in self.tokenizer("Let me try again.\n\n").input_ids])
            

            if "llama" in self.args.model_path.lower():
                str_check_token_ids = str_check_token_ids.replace("128000_", "")
                str_retry_token_ids = str_retry_token_ids.replace("128000_", "")
            
            #  split by check and retry tokens
            split_texts = re.split(f"({str_check_token_ids}|{str_retry_token_ids})", str_response_token_ids)
            
            # merge consecutive check or retry tokens
            new_split_texts = []
            last_split_token = None
            current_text = ""
            for text in split_texts:
                if text == str_check_token_ids or text == str_retry_token_ids:
                    if last_split_token == text:
                        current_text = current_text + text
                    else:
                        new_split_texts.append(current_text)
                        new_split_texts.append(text)
                        current_text = ""
                        last_split_token = text
                else:
                    current_text = current_text + text

            if current_text:
                new_split_texts.append(current_text)
            
            assert "".join(new_split_texts) == str_response_token_ids, f"Token id not match"
            
            split_texts = new_split_texts
            
            
            #  get round1 result
            round1_res_ids = split_texts[0]
            round1_tokenized = [int(token) for token in round1_res_ids.strip("_").split("_") if token]  
            round1_res = self.tokenizer.decode(round1_tokenized)
            
            current_reward_, gold_answer, current_generated_answer = get_rewards([data_dict["solution"][idx//rloo_n]], [data_dict["answer"][idx//rloo_n]], [round1_res], strict=True)
            current_reward_, gold_answer, current_generated_answer = current_reward_[0], gold_answer[0], current_generated_answer[0]

            # calculate final reward 
            if use_prefix_baseline:
                baseline = self.global_prefix_sums[current_answer_prefix] / self.global_prefix_counts[current_answer_prefix] if self.global_prefix_counts[current_answer_prefix] >= 10 else 0.0
                
                current_reward = float(current_reward_) - baseline
                
                prefix_sums[current_answer_prefix] += current_reward_
                prefix_counts[current_answer_prefix] += 1
                current_answer_prefix += f"R{current_reward_}"
            elif use_baseline:
                baseline = self.global_position_sums[0] / self.global_position_counts[0] if self.global_position_counts[0] >= self.args.batch_size else 0.0
                current_reward = float(current_reward_) - baseline
            elif use_level_baseline:
                level = all_levels[idx]   
                level_key = level.item() if isinstance(level, torch.Tensor) else level   
                baseline = self.global_position_sums[level_key][0] / self.global_position_counts[level_key][0] if self.global_position_counts[level_key][0] >= self.args.batch_size else 0.0
                current_reward = float(current_reward_) - baseline
            else:
                current_reward = current_reward_
                
            split_rewards.append(current_reward_)
            round1_rewards.append(current_reward_)
            generated_answers.append(current_generated_answer)
             
            
            current_start_token_ids = response_token_ids[:input_length+len(round1_tokenized)]
            current_token_rewards = [0] * input_length + [current_reward] * len(round1_tokenized)
            current_answer_mask = [0] * input_length + [1] * len(round1_tokenized)
            current_part_idx_masks = [0] * input_length + [1] * len(round1_tokenized)
            
            current_split_token_mask = [0] * input_length + [0] * len(round1_tokenized)
            

            final_rewards.append(current_token_rewards[-1])
            current_last_reward = current_reward_
            
            last_split_token = None
            for i, subtext_ids in enumerate(split_texts[1:]):

                 
                punish_no_end = True if (no_eos_mask_round1[idx] and i == len(split_texts) - 2) else False
                
                 
                if subtext_ids == str_check_token_ids or subtext_ids == str_retry_token_ids:
                    last_split_token = subtext_ids
                    continue
                
                 
                if last_split_token == str_check_token_ids:
                    
                    current_tokenized = [int(token) for token in subtext_ids.strip("_").split("_") if token]
                    subtext = self.tokenizer.decode(current_tokenized)

                    veri_answer = get_veri_answer(subtext, is_last=(i == len(split_texts[1:]) - 1))

                    if veri_answer == "":
                        veri_reward_ = 0
                    
                    elif current_last_reward > 0 and veri_answer == "correct" or current_last_reward < 0 and veri_answer == "incorrect":
                        veri_reward_ = 1
                    else:
                        veri_reward_ = -1
                    
                     
                    if use_prefix_baseline:
                        baseline = self.global_prefix_sums[current_answer_prefix] / self.global_prefix_counts[current_answer_prefix] if self.global_prefix_counts[current_answer_prefix] >= 10 else 0.0

                        veri_reward = float(veri_reward_) - baseline
                        
                        prefix_sums[current_answer_prefix] += veri_reward_
                        prefix_counts[current_answer_prefix] += 1
                        current_answer_prefix += f"V{veri_reward_}"
                    elif use_baseline:
                        baseline = self.global_position_sums[len(split_rewards)] / self.global_position_counts[len(split_rewards)] if self.global_position_counts[len(split_rewards)] >= self.args.batch_size else 0.0
                        veri_reward = veri_reward_ - baseline
                    elif use_level_baseline:
                        level = all_levels[idx]
                        level_key = level.item() if isinstance(level, torch.Tensor) else level   
                        pos = len(split_rewards)   
                        baseline = self.global_position_sums[level_key][pos] / self.global_position_counts[level_key][pos] if self.global_position_counts[level_key][pos] >= self.args.batch_size else 0.0
                        veri_reward = veri_reward_ - baseline
                    else:
                        veri_reward = veri_reward_
                        
                     
                    if use_veri_bonus:  
                        if current_last_reward > 0 : 
                            veri_reward = veri_reward * 2
                        else:
                            veri_reward = veri_reward
                    else:
                        veri_reward = veri_reward

                    split_rewards.append(veri_reward_)
                    veri_rewards.append(veri_reward_)
                    veri_answers.append(veri_answer)

                    split_token_tokenized = [int(token) for token in last_split_token.strip("_").split("_") if token]
                    current_start_token_ids += split_token_tokenized
                    current_token_rewards += [0] * len(split_token_tokenized)
                    current_answer_mask += [1] * len(split_token_tokenized)
                    current_part_idx_masks += [0] * len(split_token_tokenized)
                    current_split_token_mask += [1] * len(split_token_tokenized)
                    
                    current_start_token_ids += current_tokenized
                    if punish_no_end:
                        current_token_rewards += [-1] * len(current_tokenized)
                         
                    else:
                        current_token_rewards += [veri_reward] * len(current_tokenized)
                         
                    current_answer_mask += [1] * len(current_tokenized)
                    current_part_idx_masks += [i+2] * len(current_tokenized)
                    current_split_token_mask += [0] * len(current_tokenized)
                    
                 
                elif last_split_token == str_retry_token_ids:
                    
                    current_tokenized = [int(token) for token in subtext_ids.strip("_").split("_") if token]
                    subtext = self.tokenizer.decode(current_tokenized)
                     
                    current_reward_, _, current_generated_answer = get_rewards([data_dict["solution"][idx//rloo_n]], [data_dict["answer"][idx//rloo_n]], [subtext], strict=True)
                    current_reward_, current_generated_answer = current_reward_[0], current_generated_answer[0]
                    
                     
                    if use_prefix_baseline:
                        baseline = self.global_prefix_sums[current_answer_prefix] / self.global_prefix_counts[current_answer_prefix] if self.global_prefix_counts[current_answer_prefix] >= 10 else 0.0
                         
                         
                        current_reward = float(current_reward_) - baseline
                        
                        prefix_sums[current_answer_prefix] += current_reward_
                        prefix_counts[current_answer_prefix] += 1
                        current_answer_prefix += f"R{current_reward_}"
                        
                    elif use_baseline:
                        baseline = self.global_position_sums[len(split_rewards)] / self.global_position_counts[len(split_rewards)] if self.global_position_counts[len(split_rewards)] >= self.args.batch_size else 0.0
                        current_reward = current_reward_ - baseline
                    elif use_level_baseline:
                        level = all_levels[idx]
                        level_key = level.item() if isinstance(level, torch.Tensor) else level   
                        pos = len(split_rewards)   
                        baseline = self.global_position_sums[level_key][pos] / self.global_position_counts[level_key][pos] if self.global_position_counts[level_key][pos] >= self.args.batch_size else 0.0
                        current_reward = current_reward_ - baseline
                    else:
                        current_reward = current_reward_
                    
                    if use_bonus:
                        current_reward = current_reward * 2 if current_reward_ != current_last_reward else current_reward  
                    else:
                        current_reward = current_reward
                    

                    split_rewards.append(current_reward_)
                    generated_answers.append(current_generated_answer)
                     
                    
                    split_token_tokenized = [int(token) for token in last_split_token.strip("_").split("_") if token]
                    current_start_token_ids += split_token_tokenized

                     
                    current_token_rewards += [0] * len(split_token_tokenized)
                    current_answer_mask += [1] * len(split_token_tokenized)
                    current_part_idx_masks += [0] * len(split_token_tokenized)
                    current_split_token_mask += [1] * len(split_token_tokenized)
                    
                     
                    current_start_token_ids += current_tokenized
                    if punish_no_end:
                        current_token_rewards += [-1] * len(current_tokenized)
                         
                    else:
                        current_token_rewards += [current_reward] * len(current_tokenized)
                         
                    current_answer_mask += [1] * len(current_tokenized)
                    current_part_idx_masks += [i+2] * len(current_tokenized)
                    current_split_token_mask += [0] * len(current_tokenized)
                    
                    current_last_reward = current_reward_
                    
                    

                else:
                    continue
            
                final_rewards.append(current_token_rewards[-1])

            
            
            batch_split_texts.append(split_texts)
             
             
            token_rewards[idx, :len(current_token_rewards)] = torch.tensor(current_token_rewards)[:token_rewards.size(1)]
            answer_masks[idx, :len(current_answer_mask)] = torch.tensor(current_answer_mask)[:answer_masks.size(1)]
            part_idx_masks[idx, :len(current_part_idx_masks)] = torch.tensor(current_part_idx_masks)[:part_idx_masks.size(1)]
            split_token_masks[idx, :len(current_split_token_mask)] = torch.tensor(current_split_token_mask)[:split_token_masks.size(1)]

             
            last_rewards.append(current_last_reward)
            batch_generated_answers.append(generated_answers)
            
            all_split_rewards.append(split_rewards)
            all_veri_answers.append(veri_answers)
            all_veri_rewards.append(veri_rewards)
            all_final_rewards.append(final_rewards)
             
            try:
                assert current_start_token_ids == response_token_ids[:len(current_start_token_ids)]
                assert len(current_token_rewards) == len(current_answer_mask)
                valid_splits.append(1)
            except:
                valid_splits.append(0)
        
            

        if use_baseline:
            print_rank_0(f"Start to calculate new position baselines")
            position_rewards_sum = defaultdict(float)
            position_counts = defaultdict(int)
            for split_rewards in all_split_rewards:
                for pos, reward in enumerate(split_rewards):
                    position_rewards_sum[pos] += reward
                    position_counts[pos] += 1
            gathered_sums = {}
            gathered_counts = {}
            max_pos = 0

            for split_rewards in all_split_rewards:
                max_pos = max(max_pos, len(split_rewards))
            local_max_pos = torch.tensor(max_pos).to(self.accelerator.device)
            global_max_pos = self.accelerator.gather(local_max_pos).max().item()

             

            local_sums = torch.zeros(global_max_pos).to(self.accelerator.device)
            local_counts = torch.zeros(global_max_pos).to(self.accelerator.device)
             
            for pos in range(global_max_pos):
                local_sums[pos] = position_rewards_sum.get(pos, 0.0)
                local_counts[pos] = position_counts.get(pos, 0)

             
            gathered_sums = self.accelerator.reduce(local_sums, "sum").detach().cpu()
            gathered_counts = self.accelerator.reduce(local_counts, "sum").detach().cpu()
             
             
            position_baselines = {}
            for pos in range(global_max_pos):
                
                if gathered_counts[pos] > 0:
                    self.global_position_counts[pos] += gathered_counts[pos].item()
                    self.global_position_sums[pos] += gathered_sums[pos].item()
                    position_baselines[pos] = self.global_position_sums[pos] / self.global_position_counts[pos]
                else:
                    position_baselines[pos] = 0

            print_rank_0(f"Position baselines: {position_baselines}")
            print_rank_0(f"Current global_position_counts: {self.global_position_counts}")

         
        if use_level_baseline:
            print_rank_0(f"Start to calculate new position baselines")
            position_rewards_sum = defaultdict(lambda: defaultdict(float))
            position_counts = defaultdict(lambda: defaultdict(float))

            for split_rewards, level in zip(all_split_rewards, all_levels):
                for pos, reward in enumerate(split_rewards):
                    level_key = level.item() if isinstance(level, torch.Tensor) else level   
                    position_rewards_sum[level_key][pos] += reward
                    position_counts[level_key][pos] += 1


            gathered_sums = {}
            gathered_counts = {}
            max_pos = 0

            max_pos = max(len(rewards) for rewards in all_split_rewards)
            local_max_pos = torch.tensor(max_pos).to(self.accelerator.device)
            global_max_pos = self.accelerator.gather(local_max_pos).max().item()
            
            level_num = 5
            local_sums = torch.zeros(level_num+1, global_max_pos).to(self.accelerator.device)
            local_counts = torch.zeros(level_num+1, global_max_pos).to(self.accelerator.device)
            for level_key in range(1, level_num+1):
                 
                 
                for pos in range(global_max_pos):
                    local_sums[level_key][pos] = position_rewards_sum[level_key].get(pos, 0.0)
                    local_counts[level_key][pos] = position_counts[level_key].get(pos, 0)
             
             
             
            gathered_sums = self.accelerator.reduce(local_sums, "sum").detach().cpu()
            gathered_counts = self.accelerator.reduce(local_counts, "sum").detach().cpu()
             
             
            for level_key in range(1, level_num+1):
                position_baselines = {}
                for pos in range(global_max_pos):
                    if gathered_counts[level_key][pos] > 0:
                         
                        self.global_position_counts[level_key][pos] = self.global_position_counts[level_key].get(pos, 0.0) + gathered_counts[level_key][pos].item()
                        self.global_position_sums[level_key][pos] = self.global_position_sums[level_key].get(pos, 0.0) + gathered_sums[level_key][pos].item()
                        position_baselines[pos] = self.global_position_sums[level_key][pos] / self.global_position_counts[level_key][pos]
                    else:
                        position_baselines[pos] = 0
                print_rank_0(f"Level {level_key} Position baselines: {position_baselines}")
                print_rank_0(f"Level {level_key} Current global_position_counts: {self.global_position_counts[level_key]}")
         
        if use_prefix_baseline:
             
            prefix_baseline = {}

            to_pop_keys = []
            for key in prefix_sums.keys():
                if key not in self.prefix2idx:
                    to_pop_keys.append(key)
            for key in to_pop_keys:
                prefix_sums.pop(key)
                prefix_counts.pop(key)
                    
            self.global_prefix_sums = self.synchronize_dict(prefix_sums, self.global_prefix_sums)
            self.global_prefix_counts = self.synchronize_dict(prefix_counts, self.global_prefix_counts)
            for prefix in self.global_prefix_sums.keys():
                prefix_baseline[prefix] = self.global_prefix_sums[prefix] / self.global_prefix_counts[prefix] if self.global_prefix_counts[prefix] >= 10 else 0.0
            
            sorted_prefix_baseline = sorted(prefix_baseline.items(), key=lambda x: self.global_prefix_counts[x[0]], reverse=True)
            print_rank_0(f"Prefix baselines: {sorted_prefix_baseline[:10]}")
            
             
            sorted_prefix_counts = sorted(self.global_prefix_counts.items(), key=lambda x: x[1], reverse=True)
            to_print = [{"prefix": prefix, "count": count, "sum": self.global_prefix_sums[prefix]} for prefix, count in sorted_prefix_counts[:10]]
            print_rank_0(f"Prefix counts: {to_print}")

        input_ids = round1_responses

         
        attention_masks = input_ids != self.tokenizer.pad_token_id

        valid_mask = torch.LongTensor(valid_splits)

        labels = deepcopy(input_ids)
         
        labels = labels.masked_fill(answer_masks == 0, -100)

        gathered_r2_rewards = self.accelerator.gather(torch.tensor(last_rewards).to(self.accelerator.device)).detach().cpu()
        gathered_r1_rewards = self.accelerator.gather(torch.tensor(round1_rewards).to(self.accelerator.device)).detach().cpu()
         
         
        veri_rewards = [r for r_list in all_veri_rewards for r in r_list]
        veri_answers = [ans for ans_list in all_veri_answers for ans in ans_list]
        veri_rewards_sum = (torch.tensor(veri_rewards) > 0).sum()
        veri_rewards_num = (torch.tensor(veri_rewards) != 0).sum()
        veri_correct_num = torch.tensor([an == "correct" for an in veri_answers]).sum()
        gathered_veri_rewards_sum = self.accelerator.gather(veri_rewards_sum.to(self.accelerator.device)).detach().cpu().sum()
        gathered_veri_rewards_num = self.accelerator.gather(veri_rewards_num.to(self.accelerator.device)).detach().cpu().sum()
        gathered_veri_correct_num = self.accelerator.gather(veri_correct_num.to(self.accelerator.device)).detach().cpu().sum()
        
        gathered_pos_num = (gathered_r2_rewards > 0).sum()
         
        gathered_total_num = len(data_dict['problem']) * rloo_n * self.args.world_size
        print_rank_0(f"round2 positive sample : {gathered_pos_num} / {gathered_total_num}")

         
         
        veri_acc = gathered_veri_rewards_sum / gathered_veri_rewards_num
        print_rank_0(f"veri positive sample : {gathered_veri_rewards_sum} / {gathered_veri_rewards_num} = {veri_acc:.2f}")
        print_rank_0(f"veri to be correct sample : {gathered_veri_correct_num} / {gathered_veri_rewards_num} = {gathered_veri_correct_num / gathered_veri_rewards_num:.2f}")
        
        gathered_r1_incorrect_num = (gathered_r1_rewards < 0).sum()
        gathered_corrected_num = ((gathered_r1_rewards < 0) & (gathered_r2_rewards > 0)).sum()
         
        print_rank_0(f"incorrect -> correct sample : {gathered_corrected_num} / {gathered_r1_incorrect_num} = {gathered_corrected_num / gathered_r1_incorrect_num:.2f}")
        
        gathered_incorrected_num  = ((gathered_r1_rewards > 0) & (gathered_r2_rewards < 0)).sum()
         
        print_rank_0(f"correct -> incorrect sample : {gathered_incorrected_num} / {gathered_total_num - gathered_r1_incorrect_num} = {gathered_incorrected_num / (gathered_total_num - gathered_r1_incorrect_num):.2f}")
        
        relative_alter_rate = gathered_corrected_num / gathered_r1_incorrect_num - gathered_incorrected_num / (gathered_total_num - gathered_r1_incorrect_num)
        
        round1_generated_answers = [batch_generated_answers[i][0] for i in range(len(batch_generated_answers))]
        round2_generated_answers = [batch_generated_answers[i][-1] for i in range(len(batch_generated_answers))]
        gathered_incorrect_alter_num = self.accelerator.gather(torch.tensor(sum([1 for r1,a1,a2 in zip(round1_rewards, round1_generated_answers, round2_generated_answers) if r1 < 0 and a2 != a1 ])).to(self.accelerator.device)).sum().detach().cpu()
        gathered_incorrect_num = (gathered_r1_rewards < 0).sum()
         
        print_rank_0(f"incorrect -> alter sample : {gathered_incorrect_alter_num} / {gathered_incorrect_num} = {gathered_incorrect_alter_num / gathered_incorrect_num:.2f}")
        
        self_check_num = sum(valid_splits)
        gathered_selfcheck_num = self.accelerator.gather(torch.tensor(self_check_num).to(self.accelerator.device)).sum().detach().cpu()
        print_rank_0(f"Valid self-check sample : {gathered_selfcheck_num} / {gathered_total_num}")
        
        gathered_rewards = self.accelerator.gather(torch.tensor(last_rewards).to(self.accelerator.device)).detach().cpu().tolist()
         
         
        all_rewards += gathered_rewards
         
        all_sample_num = all_sample_num + gathered_total_num
        
         
         
         
        gathered_no_eos_mask = self.accelerator.gather(no_eos_mask_round1.to(self.accelerator.device)).detach().cpu()
        print_rank_0(f"no_eos_mask num: {sum([1 for i in gathered_no_eos_mask.tolist() if i])}")
        
        sft_data_num = sum(sft_data_flag)
        sft_correct_num = sum([1 for r, sft in zip(last_rewards, sft_data_flag) if r > 0 and sft])
         
        sft_veri_correct_num = sum([sum([1 for v in veri if v> 0 and sft]) for sft, veri in zip( sft_data_flag, all_veri_rewards)])
        sft_veri_num = sum([sum([1 for v in veri if v != 0 and sft]) for sft, veri in zip( sft_data_flag, all_veri_rewards)])
         
        

        
        
        gathered_sft_data_num = self.accelerator.gather(torch.tensor(sft_data_num).to(self.accelerator.device)).sum().detach().cpu()
        gathered_sft_correct_num = self.accelerator.gather(torch.tensor(sft_correct_num).to(self.accelerator.device)).sum().detach().cpu()
        gathered_none_sft_correct_num = gathered_pos_num - gathered_sft_correct_num
        none_sft_acc = gathered_none_sft_correct_num / (gathered_total_num - gathered_sft_data_num)
        print_rank_0(f"SFT data num: {gathered_sft_data_num} / {gathered_total_num}")
        if gathered_sft_data_num > 0:
            print_rank_0(f"SFT correct num: {gathered_sft_correct_num} / {gathered_sft_data_num} = {gathered_sft_correct_num / gathered_sft_data_num:.2f}")
        print_rank_0(f"None SFT correct num: {gathered_none_sft_correct_num} / {gathered_total_num - gathered_sft_data_num} = {gathered_none_sft_correct_num / (gathered_total_num - gathered_sft_data_num):.2f}")
        
        gathered_sft_veri_correct_num = self.accelerator.gather(torch.tensor(sft_veri_correct_num).to(self.accelerator.device)).sum().detach().cpu()
        gathered_sft_veri_num = self.accelerator.gather(torch.tensor(sft_veri_num).to(self.accelerator.device)).sum().detach().cpu()
        gathered_none_sft_veri_correct_num = gathered_veri_rewards_sum - gathered_sft_veri_correct_num
        if gathered_sft_veri_num > 0:
            print_rank_0(f"SFT veri correct num: {gathered_sft_veri_correct_num} / {gathered_sft_veri_num} = {gathered_sft_veri_correct_num / gathered_sft_veri_num:.2f}")
        print_rank_0(f"None SFT veri correct num: {gathered_none_sft_veri_correct_num} / {gathered_veri_rewards_num - gathered_sft_veri_num} = {gathered_none_sft_veri_correct_num / (gathered_veri_rewards_num - gathered_sft_veri_num):.2f}")


         
        model.eval()
        torch.cuda.empty_cache()
         

        old_forward_logits = []
        max_forward_size = 64 // self.args.world_size
        if len(input_ids) > max_forward_size:
            
            for i in range(len(input_ids) // max_forward_size + 1):
                input_ids_ = input_ids[i*max_forward_size:(i+1)*max_forward_size]
                if input_ids_.size(0) == 0:
                    break
                attention_masks_ = attention_masks[i*max_forward_size:(i+1)*max_forward_size]
                with torch.no_grad():
                    if self.args.use_tf32_forward:
                        with torch.autocast(dtype=torch.float32, device_type="cuda"):
                            old_model_outputs = model(
                                input_ids=input_ids_.to(model.device),
                                attention_mask=attention_masks_.to(model.device)
                            )
                    else:
                        old_model_outputs = model(
                            input_ids=input_ids_.to(model.device),
                            attention_mask=attention_masks_.to(model.device)
                        )
                    old_forward_logits.append(old_model_outputs.logits.detach().cpu())
                     
                    del old_model_outputs
                     
            if old_forward_logits:
                old_forward_logits = torch.cat(old_forward_logits, dim=0)
            else:
                old_forward_logits = torch.tensor([])
                    
        else:
            with torch.no_grad():
                old_model_outputs = model(
                    input_ids=input_ids.to(model.device),
                    attention_mask=attention_masks.to(model.device)
                )
                old_forward_logits = old_model_outputs.logits.detach().cpu()
                del old_model_outputs    
        
        model.train()

        data_collator_dict = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_masks,
            "token_rewards": token_rewards,
            "instance_rewards": torch.tensor(last_rewards),
            "weights": torch.tensor([1. if valid else 0. for valid in valid_splits  ]),
            "answer_masks": answer_masks,
            "part_idx_masks": part_idx_masks,
            "split_token_masks": split_token_masks,
            "sft_mask": torch.tensor([0. for _ in valid_splits]),
            "valid_mask": valid_mask,
            "old_forward_logits": old_forward_logits,
        }
        
        with open(self.args.dump_data_path, "a") as f:
            for idx, (response1) in enumerate(zip(round1_decoded_responses)):

                if self.args.dump_data_path is not None:
                    filter_map = lambda x: x.item() if type(x) == torch.Tensor else x
                    dump_data = {key: filter_map(data_dict[key][idx//rloo_n]) for key in data_dict.keys()}
                    dump_data["round_1_instruction"] = filter_map(seq[0]["content"])
                    dump_data["split_texts"] = filter_map(batch_split_texts[idx])
                    dump_data["full_response"] = filter_map(response1)
                     
                    dump_data["all_split_rewards"] = filter_map(all_split_rewards[idx])
                    dump_data["all_veri_answers"] = filter_map(all_veri_answers[idx])
                    dump_data["all_veri_rewards"] = filter_map(all_veri_rewards[idx])
                    dump_data["all_final_rewards"] = filter_map(all_final_rewards[idx])
                    dump_data["extracted_answers"] = filter_map(batch_generated_answers[idx])
                    dump_data["gold_extracted_answer"] = filter_map(data_dict["answer"][idx//rloo_n])
                    
                     
                    try:
                        f.write(json.dumps(dump_data) + "\n")
                    except:
                        print("dump data error")
                
        torch.cuda.empty_cache()
        
        return ExpMinDataset(data_collator_dict), all_rewards, all_sample_num, relative_alter_rate, none_sft_acc



    def generate_experience_incot_splitmask_rloo(self, model, data_dict, all_rewards, all_sample_num, stage="stage1", use_bonus=False, use_veri_bonus=False, use_baseline=False, use_level_baseline=True, rloo_n=1):
        '''
        raw_prompt_list: list of string (input problems)
        '''
        
        self.round_1_instruction_template = "Please reason step by step, and put your final answer within \\boxed{}."
         
        model.eval()  
        self.tokenizer.padding_side = "left"
         
         
        round1_input_ids_list = []

        for line in data_dict["problem"]:

            seq = [{"role": "system", "content": self.round_1_instruction_template},
                       {"role": "user", "content": line}]
            
            tokenized_inputs = self.tokenizer.apply_chat_template(seq, return_tensors="pt", add_generation_prompt=True, tokenize=True)[0]
             
             
            round1_input_ids_list.append(tokenized_inputs)

        
        round1_input_ids = padding_any(round1_input_ids_list, padding_strategy="longest", max_length=self.tokenizer.model_max_length, padding_side="left", pad_token=self.tokenizer.pad_token_id)

        with torch.no_grad():
            round1_input_ids = round1_input_ids.to(model.device)
            input_length = round1_input_ids.size(1)
            round1_attention_masks = (round1_input_ids != self.tokenizer.pad_token_id).to(model.dtype)  
             
            if self.args.use_tf32_forward:
                with torch.autocast(dtype=torch.float32, device_type="cuda"):
                    round1_responses_ = model.generate(round1_input_ids, attention_mask=round1_attention_masks,max_length=self.tokenizer.model_max_length,max_new_tokens=None,
                                        num_return_sequences=rloo_n, temperature=0.7, top_p=1.0, do_sample=True, synced_gpus=True,
                                        pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
            else:
                round1_responses_ = model.generate(round1_input_ids, attention_mask=round1_attention_masks,max_length=self.tokenizer.model_max_length,max_new_tokens=None,
                                        num_return_sequences=rloo_n, temperature=0.7, top_p=1.0, do_sample=True, synced_gpus=True,
                                        pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)

            no_eos_mask_round1 = ((round1_responses_[:,-1] != self.tokenizer.eos_token_id) & (round1_responses_[:,-1] != self.tokenizer.pad_token_id)).detach().cpu()
            round1_responses_ = round1_responses_.detach().cpu()
            round1_input_ids = round1_input_ids.detach().cpu()
            round1_responses = torch.full((round1_responses_.size(0), self.tokenizer.model_max_length), self.tokenizer.pad_token_id)
            round1_responses[:, :round1_responses_.size(1)] = round1_responses_
         
         
        input_length = round1_input_ids.size(1)
        round1_decoded_responses = self.tokenizer.batch_decode(round1_responses[:, input_length:], skip_special_tokens=True) 
                                     


        torch.cuda.empty_cache()
        
        valid_splits = []
        last_rewards = []
        round1_rewards = []
        
        

        batch_split_texts = []
        batch_generated_answers = []
        
        all_split_rewards = []
        all_veri_answers = []
        all_veri_rewards = []
        all_final_rewards = []
        all_levels = []
         
        
        sft_data_flag = []
        
         
         
        token_rewards = torch.zeros_like(round1_responses)
        answer_masks = torch.zeros_like(round1_responses)
         
        split_token_masks = torch.zeros_like(round1_responses)
        
         
        part_idx_masks = torch.zeros_like(round1_responses)
        
         
        for idx, (response_text, response_token_ids) in enumerate(zip(round1_decoded_responses, round1_responses)):
            
            response_token_ids = response_token_ids.tolist()
            sft_data_flag.append(data_dict["is_sft"][idx//rloo_n])
            all_levels.append(data_dict["level"][idx//rloo_n])
            
            split_rewards = []
            veri_rewards = []
            veri_answers = []
            final_rewards = []

            generated_answers = []

            str_response_token_ids = "_".join([str(token) for token in response_token_ids[input_length:]])
            str_response_token_ids = str_response_token_ids.split(str(self.tokenizer.eos_token_id))[0].strip("_")
             
             
            str_check_token_ids = "_".join([str(token) for token in self.tokenizer("Wait,").input_ids])
            str_retry_token_ids = "_".join([str(token) for token in self.tokenizer("Let me try again.\n\n").input_ids])
            
            if "llama" in self.args.model_path.lower():
                str_check_token_ids = str_check_token_ids.replace("128000_", "")
                str_retry_token_ids = str_retry_token_ids.replace("128000_", "")
            split_texts = re.split(f"({str_check_token_ids}|{str_retry_token_ids})", str_response_token_ids)

            round1_res_ids = split_texts[0]
            round1_tokenized = [int(token) for token in round1_res_ids.strip("_").split("_") if token]  
            round1_res = self.tokenizer.decode(round1_tokenized)
            
             
            current_reward_, gold_answer, current_generated_answer = get_rewards([data_dict["solution"][idx//rloo_n]], [data_dict["answer"][idx//rloo_n]], [round1_res], strict=True)
            
            current_reward_, gold_answer, current_generated_answer = current_reward_[0], gold_answer[0], current_generated_answer[0]

            if use_baseline:
                baseline = self.global_position_sums[0] / self.global_position_counts[0] if self.global_position_counts[0] >= self.args.batch_size else 0.0
                current_reward = float(current_reward_) - baseline
            elif use_level_baseline:
                level = all_levels[idx//rloo_n]   
                level_key = level.item() if isinstance(level, torch.Tensor) else level   
                baseline = self.global_position_sums[level_key][0] / self.global_position_counts[level_key][0] if self.global_position_counts[level_key][0] >= self.args.batch_size else 0.0
                current_reward = float(current_reward_) - baseline
            else:
                current_reward = current_reward_
            split_rewards.append(current_reward_)
            round1_rewards.append(current_reward_)
            generated_answers.append(current_generated_answer)
             
            
            current_start_token_ids = response_token_ids[:input_length+len(round1_tokenized)]
            if stage == "stage1":  
                current_token_rewards = [0] * input_length + [0] * len(round1_tokenized)
            else:
                 
                current_token_rewards = [0] * input_length + [current_reward] * len(round1_tokenized)
            current_answer_mask = [0] * input_length + [1] * len(round1_tokenized)
            current_part_idx_masks = [0] * input_length + [1] * len(round1_tokenized)
            
            current_split_token_mask = [0] * input_length + [0] * len(round1_tokenized)
            
             

            final_rewards.append(current_token_rewards[-1])
            current_last_reward = current_reward_
            last_split_token = None
            for i, subtext_ids in enumerate(split_texts[1:]):

                punish_no_end = True if (no_eos_mask_round1[idx] and i == len(split_texts) - 2) else False
                
                # check token
                if subtext_ids == str_check_token_ids or subtext_ids == str_retry_token_ids:
                    last_split_token = subtext_ids
                    continue
                
                # check part
                if last_split_token == str_check_token_ids:
                    
                    current_tokenized = [int(token) for token in subtext_ids.strip("_").split("_") if token]
                    subtext = self.tokenizer.decode(current_tokenized)
                    
                    # get veri answer
                    veri_answer = get_veri_answer(subtext, is_last=(i == len(split_texts[1:]) - 1))
                    
                    # get veri reward
                    if veri_answer == "":
                        veri_reward_ = 0
                    elif current_last_reward > 0 and veri_answer == "correct" or current_last_reward < 0 and veri_answer == "incorrect":
                        veri_reward_ = 1
                    else:
                        veri_reward_ = -1
                    
                    # get final veri reward
                    if use_baseline:
                        baseline = self.global_position_sums[len(split_rewards)] / self.global_position_counts[len(split_rewards)] if self.global_position_counts[len(split_rewards)] >= self.args.batch_size else 0.0
                        veri_reward = veri_reward_ - baseline
                    elif use_level_baseline:
                        level = all_levels[idx//rloo_n]
                        level_key = level.item() if isinstance(level, torch.Tensor) else level   
                        pos = len(split_rewards)   
                        baseline = self.global_position_sums[level_key][pos] / self.global_position_counts[level_key][pos] if self.global_position_counts[level_key][pos] >= self.args.batch_size else 0.0
                        veri_reward = veri_reward_ - baseline
                    else:
                        veri_reward = veri_reward_
                        
                    if use_veri_bonus:  
                        if current_last_reward > 0:
                            veri_reward = veri_reward * 2
                        else:
                            veri_reward = veri_reward
                    else:
                        veri_reward = veri_reward

                    split_rewards.append(veri_reward_)
                    veri_rewards.append(veri_reward_)
                    veri_answers.append(veri_answer)
                     

                    split_token_tokenized = [int(token) for token in last_split_token.strip("_").split("_") if token]
                    current_start_token_ids += split_token_tokenized
                    current_token_rewards += [0] * len(split_token_tokenized)
                    current_answer_mask += [1] * len(split_token_tokenized)
                    current_part_idx_masks += [0] * len(split_token_tokenized)
                    current_split_token_mask += [1] * len(split_token_tokenized)
                    
                    current_start_token_ids += current_tokenized
                    if punish_no_end:
                        current_token_rewards += [-1] * len(current_tokenized)
                    else:
                        current_token_rewards += [veri_reward] * len(current_tokenized)
                         
                    current_answer_mask += [1] * len(current_tokenized)
                    current_part_idx_masks += [i+2] * len(current_tokenized)
                    current_split_token_mask += [0] * len(current_tokenized)

                # retry part
                elif last_split_token == str_retry_token_ids:
                    
                    current_tokenized = [int(token) for token in subtext_ids.strip("_").split("_") if token]
                    subtext = self.tokenizer.decode(current_tokenized)
                     
                    current_reward_, _, current_generated_answer = get_rewards([data_dict["solution"][idx//rloo_n]], [data_dict["answer"][idx//rloo_n]], [subtext], strict=True)
                    current_reward_, current_generated_answer = current_reward_[0], current_generated_answer[0]
                    
                    # get final reward
                    if use_baseline:
                        baseline = self.global_position_sums[len(split_rewards)] / self.global_position_counts[len(split_rewards)] if self.global_position_counts[len(split_rewards)] >= self.args.batch_size else 0.0
                        current_reward = current_reward_ - baseline
                    elif use_level_baseline:
                        level = all_levels[idx//rloo_n]
                        level_key = level.item() if isinstance(level, torch.Tensor) else level   
                        pos = len(split_rewards)   
                        baseline = self.global_position_sums[level_key][pos] / self.global_position_counts[level_key][pos] if self.global_position_counts[level_key][pos] >= self.args.batch_size else 0.0
                        current_reward = current_reward_ - baseline
                    else:
                        current_reward = current_reward_
                    if use_bonus:
                        current_reward = current_reward * 2 if current_reward_ != current_last_reward else current_reward  
                    else:
                        current_reward = current_reward

                    split_rewards.append(current_reward_)
                    generated_answers.append(current_generated_answer)
   
                    split_token_tokenized = [int(token) for token in last_split_token.strip("_").split("_") if token]
                    current_start_token_ids += split_token_tokenized

                    current_token_rewards += [0] * len(split_token_tokenized)
                    current_answer_mask += [1] * len(split_token_tokenized)
                    current_part_idx_masks += [0] * len(split_token_tokenized)
                    current_split_token_mask += [1] * len(split_token_tokenized)
                    
                     
                    current_start_token_ids += current_tokenized
                    
                    if punish_no_end:
                        current_token_rewards += [-1] * len(current_tokenized)
                    else:
                        current_token_rewards += [current_reward] * len(current_tokenized)
                         
                    current_answer_mask += [1] * len(current_tokenized)
                    current_part_idx_masks += [i+2] * len(current_tokenized)
                    current_split_token_mask += [0] * len(current_tokenized)

                    current_last_reward = current_reward_
                
                else:
                    continue

                final_rewards.append(current_token_rewards[-1])

            batch_split_texts.append(split_texts)
            token_rewards[idx, :len(current_token_rewards)] = torch.tensor(current_token_rewards)[:token_rewards.size(1)]
            answer_masks[idx, :len(current_answer_mask)] = torch.tensor(current_answer_mask)[:answer_masks.size(1)]
            part_idx_masks[idx, :len(current_part_idx_masks)] = torch.tensor(current_part_idx_masks)[:part_idx_masks.size(1)]
            split_token_masks[idx, :len(current_split_token_mask)] = torch.tensor(current_split_token_mask)[:split_token_masks.size(1)]
             
            last_rewards.append(current_last_reward)
            batch_generated_answers.append(generated_answers)
            
            all_split_rewards.append(split_rewards)
            all_veri_answers.append(veri_answers)
            all_veri_rewards.append(veri_rewards)
            all_final_rewards.append(final_rewards)
             
            try:
                assert current_start_token_ids == response_token_ids[:len(current_start_token_ids)]
                assert len(current_token_rewards) == len(current_answer_mask) == len(current_part_idx_masks) 
                valid_splits.append(1)
            except:
                valid_splits.append(0)
        
            

        if use_baseline:
            print_rank_0(f"Start to calculate new position baselines")
            position_rewards_sum = defaultdict(float)
            position_counts = defaultdict(int)
            for split_rewards in all_split_rewards:
                for pos, reward in enumerate(split_rewards):
                    position_rewards_sum[pos] += reward
                    position_counts[pos] += 1
            gathered_sums = {}
            gathered_counts = {}
            max_pos = 0

            for split_rewards in all_split_rewards:
                max_pos = max(max_pos, len(split_rewards))
            local_max_pos = torch.tensor(max_pos).to(self.accelerator.device)
            global_max_pos = self.accelerator.gather(local_max_pos).max().item()

            level_key = level.item() if isinstance(level, torch.Tensor) else level

            local_sums = torch.zeros(global_max_pos).to(self.accelerator.device)
            local_counts = torch.zeros(global_max_pos).to(self.accelerator.device)
             
            for pos in range(global_max_pos):
                local_sums[pos] = position_rewards_sum[level_key].get(pos, 0.0)
                local_counts[pos] = position_counts[level_key].get(pos, 0)

             
            gathered_sums = self.accelerator.reduce(local_sums, "sum").detach().cpu()
            gathered_counts = self.accelerator.reduce(local_counts, "sum").detach().cpu()
             
             
            position_baselines = {}
            for pos in range(global_max_pos):
                
                if gathered_counts[pos] > 0:
                    self.global_position_counts[pos] += gathered_counts[pos].item()
                    self.global_position_sums[pos] += gathered_sums[pos].item()
                    position_baselines[pos] = self.global_position_sums[pos] / self.global_position_counts[pos]
                else:
                    position_baselines[pos] = 0

            print_rank_0(f"Position baselines: {position_baselines}")
            print_rank_0(f"Current global_position_counts: {self.global_position_counts}")

         
        if use_level_baseline:
            print_rank_0(f"Start to calculate new position baselines")
            position_rewards_sum = defaultdict(lambda: defaultdict(float))
            position_counts = defaultdict(lambda: defaultdict(float))

            for split_rewards, level in zip(all_split_rewards, all_levels):
                for pos, reward in enumerate(split_rewards):
                    level_key = level.item() if isinstance(level, torch.Tensor) else level   
                    position_rewards_sum[level_key][pos] += reward
                    position_counts[level_key][pos] += 1


            gathered_sums = {}
            gathered_counts = {}
            max_pos = 0

            max_pos = max(len(rewards) for rewards in all_split_rewards)
            local_max_pos = torch.tensor(max_pos).to(self.accelerator.device)
            global_max_pos = self.accelerator.gather(local_max_pos).max().item()
            
            for level in set(all_levels):
                level_key = level.item() if isinstance(level, torch.Tensor) else level

                local_sums = torch.zeros(global_max_pos).to(self.accelerator.device)
                local_counts = torch.zeros(global_max_pos).to(self.accelerator.device)
                 
                for pos in range(global_max_pos):
                    local_sums[pos] = position_rewards_sum[level_key].get(pos, 0.0)
                    local_counts[pos] = position_counts[level_key].get(pos, 0)

                 
                gathered_sums = self.accelerator.reduce(local_sums, "sum").detach().cpu()
                gathered_counts = self.accelerator.reduce(local_counts, "sum").detach().cpu()
                
                position_baselines = {}
                for pos in range(global_max_pos):
                    if gathered_counts[pos] > 0:
                         
                        self.global_position_counts[level_key][pos] = self.global_position_counts[level_key].get(pos, 0.0) + gathered_counts[pos].item()
                        self.global_position_sums[level_key][pos] = self.global_position_sums[level_key].get(pos, 0.0) + gathered_sums[pos].item()
                        position_baselines[pos] = self.global_position_sums[level_key][pos] / self.global_position_counts[level_key][pos]
                    else:
                        position_baselines[pos] = 0
                print_rank_0(f"Level {level_key} Position baselines: {position_baselines}")
                print_rank_0(f"Level {level_key} Current global_position_counts: {self.global_position_counts[level_key]}")
         

     




         
        
         
        input_ids = round1_responses

         
        attention_masks = input_ids != self.tokenizer.pad_token_id

        valid_mask = torch.LongTensor(valid_splits)

        labels = deepcopy(input_ids)
         
        labels = labels.masked_fill(answer_masks == 0, -100).long()


        gathered_r2_rewards = self.accelerator.gather(torch.tensor(last_rewards).to(self.accelerator.device)).detach().cpu()
        gathered_r1_rewards = self.accelerator.gather(torch.tensor(round1_rewards).to(self.accelerator.device)).detach().cpu()
         
         
        veri_rewards = [r for r_list in all_veri_rewards for r in r_list]
        veri_answers = [ans for ans_list in all_veri_answers for ans in ans_list]
        veri_rewards_sum = (torch.tensor(veri_rewards) > 0).sum()
        veri_rewards_num = (torch.tensor(veri_rewards) != 0).sum()
        veri_correct_num = torch.tensor([an == "correct" for an in veri_answers]).sum()
        gathered_veri_rewards_sum = self.accelerator.gather(veri_rewards_sum.to(self.accelerator.device)).detach().cpu().sum()
        gathered_veri_rewards_num = self.accelerator.gather(veri_rewards_num.to(self.accelerator.device)).detach().cpu().sum()
        gathered_veri_correct_num = self.accelerator.gather(veri_correct_num.to(self.accelerator.device)).detach().cpu().sum()
        
        gathered_pos_num = (gathered_r2_rewards > 0).sum()
         
        gathered_total_num = len(data_dict['problem']) * rloo_n * self.args.world_size
        print_rank_0(f"round2 positive sample : {gathered_pos_num} / {gathered_total_num}")
         
         
        veri_acc = gathered_veri_rewards_sum / gathered_veri_rewards_num
        print_rank_0(f"veri positive sample : {gathered_veri_rewards_sum} / {gathered_veri_rewards_num} = {veri_acc:.2f}")
        print_rank_0(f"veri to be correct sample : {gathered_veri_correct_num} / {gathered_veri_rewards_num} = {gathered_veri_correct_num / gathered_veri_rewards_num:.2f}")
        
        gathered_r1_incorrect_num = (gathered_r1_rewards < 0).sum()
        gathered_corrected_num = ((gathered_r1_rewards < 0) & (gathered_r2_rewards > 0)).sum()
         
        print_rank_0(f"incorrect -> correct sample : {gathered_corrected_num} / {gathered_r1_incorrect_num} = {gathered_corrected_num / gathered_r1_incorrect_num:.2f}")
        
        gathered_incorrected_num  = ((gathered_r1_rewards > 0) & (gathered_r2_rewards < 0)).sum()
         
        print_rank_0(f"correct -> incorrect sample : {gathered_incorrected_num} / {gathered_total_num - gathered_r1_incorrect_num} = {gathered_incorrected_num / (gathered_total_num - gathered_r1_incorrect_num):.2f}")
        
        relative_alter_rate = gathered_corrected_num / gathered_r1_incorrect_num - gathered_incorrected_num / (gathered_total_num - gathered_r1_incorrect_num)
        
        round1_generated_answers = [batch_generated_answers[i][0] for i in range(len(batch_generated_answers))]
        round2_generated_answers = [batch_generated_answers[i][-1] for i in range(len(batch_generated_answers))]
        gathered_incorrect_alter_num = self.accelerator.gather(torch.tensor(sum([1 for r1,a1,a2 in zip(round1_rewards, round1_generated_answers, round2_generated_answers) if r1 < 0 and a2 != a1 ])).to(self.accelerator.device)).sum().detach().cpu()
        gathered_incorrect_num = (gathered_r1_rewards < 0).sum()
         
        print_rank_0(f"incorrect -> alter sample : {gathered_incorrect_alter_num} / {gathered_incorrect_num} = {gathered_incorrect_alter_num / gathered_incorrect_num:.2f}")
        
        self_check_num = sum(valid_splits)
        gathered_selfcheck_num = self.accelerator.gather(torch.tensor(self_check_num).to(self.accelerator.device)).sum().detach().cpu()
        print_rank_0(f"Valid self-check sample : {gathered_selfcheck_num} / {gathered_total_num}")
        
        gathered_rewards = self.accelerator.gather(torch.tensor(last_rewards).to(self.accelerator.device)).detach().cpu().tolist()
         
         
        all_rewards += gathered_rewards
         
        all_sample_num = all_sample_num + gathered_total_num
        
         
         
         
        gathered_no_eos_mask = self.accelerator.gather(no_eos_mask_round1.to(self.accelerator.device)).detach().cpu()
        print_rank_0(f"no_eos_mask num: {sum([1 for i in gathered_no_eos_mask.tolist() if i])}")
        
        sft_data_num = sum(sft_data_flag)
        sft_correct_num = sum([1 for r, sft in zip(last_rewards, sft_data_flag) if r > 0 and sft])
         
        sft_veri_correct_num = sum([sum([1 for v in veri if v> 0 and sft]) for sft, veri in zip( sft_data_flag, all_veri_rewards)])
        sft_veri_num = sum([sum([1 for v in veri if v != 0 and sft]) for sft, veri in zip( sft_data_flag, all_veri_rewards)])

        gathered_sft_data_num = self.accelerator.gather(torch.tensor(sft_data_num).to(self.accelerator.device)).sum().detach().cpu()
        gathered_sft_correct_num = self.accelerator.gather(torch.tensor(sft_correct_num).to(self.accelerator.device)).sum().detach().cpu()
        gathered_none_sft_correct_num = gathered_pos_num - gathered_sft_correct_num
        none_sft_acc = gathered_none_sft_correct_num / (gathered_total_num - gathered_sft_data_num)
        print_rank_0(f"SFT data num: {gathered_sft_data_num} / {gathered_total_num}")
        if gathered_sft_data_num != 0:
            print_rank_0(f"SFT correct num: {gathered_sft_correct_num} / {gathered_sft_data_num} = {gathered_sft_correct_num / gathered_sft_data_num:.2f}")
        print_rank_0(f"None SFT correct num: {gathered_none_sft_correct_num} / {gathered_total_num - gathered_sft_data_num} = {gathered_none_sft_correct_num / (gathered_total_num - gathered_sft_data_num):.2f}")
        
        gathered_sft_veri_correct_num = self.accelerator.gather(torch.tensor(sft_veri_correct_num).to(self.accelerator.device)).sum().detach().cpu()
        gathered_sft_veri_num = self.accelerator.gather(torch.tensor(sft_veri_num).to(self.accelerator.device)).sum().detach().cpu()
        gathered_none_sft_veri_correct_num = gathered_veri_rewards_sum - gathered_sft_veri_correct_num
        if gathered_sft_veri_num != 0:
            print_rank_0(f"SFT veri correct num: {gathered_sft_veri_correct_num} / {gathered_sft_veri_num} = {gathered_sft_veri_correct_num / gathered_sft_veri_num:.2f}")
        print_rank_0(f"None SFT veri correct num: {gathered_none_sft_veri_correct_num} / {gathered_veri_rewards_num - gathered_sft_veri_num} = {gathered_none_sft_veri_correct_num / (gathered_veri_rewards_num - gathered_sft_veri_num):.2f}")


         
        model.eval()
        torch.cuda.empty_cache()
        

        
        old_forward_logits = []
        ref_forward_logits = []
        max_forward_size = 64 // self.args.world_size
        if len(input_ids) > max_forward_size:
            
            for i in range(len(input_ids) // max_forward_size + 1):
                input_ids_ = input_ids[i*max_forward_size:(i+1)*max_forward_size]
                if input_ids_.size(0) == 0:
                    break
                attention_masks_ = attention_masks[i*max_forward_size:(i+1)*max_forward_size]
                with torch.no_grad():
                    if self.args.use_tf32_forward:
                        with torch.autocast(dtype=torch.float32, device_type="cuda"):
                            old_model_outputs = model(
                                input_ids=input_ids_.to(model.device),
                                attention_mask=attention_masks_.to(model.device)
                            )
                    else:
                        old_model_outputs = model(
                                input_ids=input_ids_.to(model.device),
                                attention_mask=attention_masks_.to(model.device)
                            )
                    old_forward_logits.append(old_model_outputs.logits.detach().cpu().to(model.dtype))
                     
                    del old_model_outputs    
            old_forward_logits = torch.cat(old_forward_logits, dim=0)
             
                    
        else:
            with torch.no_grad():
                 
                old_model_outputs = model(
                    input_ids=input_ids.to(model.device),
                    attention_mask=attention_masks.to(model.device)
                )
                old_forward_logits = old_model_outputs.logits.detach().cpu().to(model.dtype)
                del old_model_outputs   
                 

        instance_rewards = torch.tensor(last_rewards).to(model.dtype)
        
        
        if rloo_n > 1:
            all_final_rewards = []
            baselines = []
            for problem_id in range(len(data_dict["problem"])):
                example_token_rewards = token_rewards[problem_id*rloo_n:(problem_id+1)*rloo_n]
                example_part_idx_masks = part_idx_masks[problem_id*rloo_n:(problem_id+1)*rloo_n]
                
                final_rewards = [[None for i in range(example_part_idx_masks.max()+1) ] for j in range(rloo_n)]
                for part_idx in range(1, example_part_idx_masks.max()+1):
                    part_mask = (example_part_idx_masks == part_idx )
                    if part_mask.sum() == 0:
                        continue
                    part_rewards = torch.zeros(rloo_n)
                    for i in range(rloo_n):
                        if part_mask[i].sum() == 0:
                            continue
                        part_rewards[i] = (example_token_rewards[i] * part_mask[i]).sum() / (part_mask[i].sum()+1e-10)  
                    
                    for i in range(rloo_n):
                         
                        if part_mask[i].sum(-1) == 0:
                            continue
                        
                        elif (part_rewards!=0).sum() < 2:
                            final_rewards[i][part_idx] = part_rewards[i]
                            continue
                        
                        baseline = (sum(part_rewards) - part_rewards[i]) / ((part_rewards!=0).sum() -1)
                        rloo_reward =  part_rewards[i] - baseline + part_rewards[i]
                         
                        try:
                            token_rewards[problem_id*rloo_n+i] = token_rewards[problem_id*rloo_n+i].masked_fill(part_mask[i], rloo_reward.float())
                            final_rewards[i][part_idx] = rloo_reward
                        except:
                            print_rank_0(f"type type(token_rewards[problem_id*rloo_n+i]): {type(token_rewards[problem_id*rloo_n+i])}")
                            print_rank_0(f"type part_rewards: {type(part_rewards)}")
                            print_rank_0(f"type part_mask[i]: {type(part_mask[i])}")
                            print_rank_0(f"type rloo_reward: {type(rloo_reward)}")
                            print_rank_0(f"size token_rewards[problem_id*rloo_n+i]: {token_rewards[problem_id*rloo_n+i].size()}")
                            print_rank_0(f"size rloo_reward: {rloo_reward.size()}")
                            print_rank_0(f"size part_mask[i]: {part_mask[i].size()}")
                            print_rank_0(f"token_rewards[problem_id*rloo_n+i]: {token_rewards[problem_id*rloo_n+i]}")
                            print_rank_0(f"part_mask[i]: {part_mask[i]}")
                            print_rank_0(f"rloo_reward: {rloo_reward}")
                
                all_final_rewards.extend(final_rewards)
                baselines.append(baseline)
                            
            

            
            if instance_rewards.std() == 0:
                print_rank_0(f"instance_rewards.std() == 0")
            
            # instance-level reward
            instance_rewards = torch.clamp(instance_rewards, -10., 10.)

            instance_rewards = instance_rewards.view(len(data_dict["problem"]), rloo_n)
            instance_baseline = (instance_rewards.sum(-1).view(-1, 1).repeat(1, rloo_n) - instance_rewards) / (rloo_n - 1)
            instance_rloo_rewards = instance_rewards - instance_baseline
            instance_rloo_rewards = instance_rloo_rewards.view(-1)
            instance_rewards = instance_rloo_rewards
            

        model.train()

        data_collator_dict = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_masks,
            "token_rewards": token_rewards,
            "instance_rewards": instance_rewards,
            "weights": (instance_rewards != 0).to(model.dtype), 
            "answer_masks": answer_masks,
            "part_idx_masks": part_idx_masks,
            "split_token_masks": split_token_masks,
            "sft_mask": torch.tensor([0. for _ in valid_splits]),
            "valid_mask": valid_mask,
            "old_forward_logits": old_forward_logits,
        }
        
        with open(self.args.dump_data_path, "a") as f:
            for idx, response1 in enumerate(round1_decoded_responses):
                
                 
                 
                 
                 

                 
                 
                 
                 
                if self.args.dump_data_path is not None:
                    filter_map = lambda x: x.item() if type(x) == torch.Tensor else x
                    dump_data = {key: filter_map(data_dict[key][idx//rloo_n]) for key in data_dict.keys()}
                    dump_data["round_1_instruction"] = filter_map(seq[0]["content"])
                    dump_data["split_texts"] = filter_map(batch_split_texts[idx])
                    dump_data["full_response"] = filter_map(response1)
                    dump_data["all_split_rewards"] = filter_map(all_split_rewards[idx])
                    dump_data["instance_reward"] = filter_map(instance_rewards[idx]) if rloo_n == 1 else filter_map(instance_rloo_rewards[idx])
                    dump_data["instance_reward_final"] = filter_map(instance_rewards[idx]) 
                    dump_data["all_veri_answers"] = filter_map(all_veri_answers[idx])
                    dump_data["all_veri_rewards"] = filter_map(all_veri_rewards[idx])
                    dump_data["all_final_rewards"] = filter_map(all_final_rewards[idx])
                    dump_data["extracted_answers"] = filter_map(batch_generated_answers[idx])
                    dump_data["gold_extracted_answer"] = filter_map(data_dict["answer"][idx//rloo_n])
                    
                    for key in dump_data.keys():
                        dump_data[key] = filter_map(dump_data[key])
                        if type(dump_data[key]) == list:
                            dump_data[key] = [filter_map(x) for x in dump_data[key]]
                     
                    try:
                        f.write(json.dumps(dump_data) + "\n")
                    except Exception as e:
                        print_rank_0("dump data error: ", e)
                        print_rank_0("dump_data: ", dump_data)
                
        torch.cuda.empty_cache()
        
        return ExpMinDataset(data_collator_dict), all_rewards, all_sample_num, relative_alter_rate, none_sft_acc



    def compute_loss_splitpart_inkl_sentence(self, model, inputs, sft_data=None, return_outputs=False, use_sft_loss=False, stage="stage1"):

        if self.args.use_tf32_forward:
            with torch.autocast(dtype=torch.float32, device_type="cuda"):
                model_outputs = model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    labels=inputs['labels']
                )
        else:
            model_outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                labels=inputs['labels']
            )

        logprobs, mask = compute_lm_loglikeli(model_outputs.logits, inputs['labels'])  

         
        with torch.no_grad():
            self.ref_model.eval()
            if self.args.use_tf32_forward:
                with torch.autocast(dtype=torch.float32, device_type="cuda"):
                    ref_model_outputs = self.ref_model(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask']
                    )
            else:
                ref_model_outputs = self.ref_model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask']
                )
            ref_logprobs, _ = compute_lm_loglikeli(ref_model_outputs.logits, inputs['labels'])
            
            old_logprobs, old_mask = compute_lm_loglikeli(inputs["old_forward_logits"], inputs['labels'])

            kl_div = self.compute_kl_divergence(old_logprobs, ref_logprobs, kl_penalty="kl")

            token_rewards = inputs['token_rewards'][..., 1:].contiguous()
            

        total_loss = 0
        num_valid_parts = 0
        inputs['part_idx_masks'] = inputs['part_idx_masks'][..., 1:].contiguous()
        existing_parts = torch.unique(inputs['part_idx_masks'])
        part_num = torch.zeros(inputs['part_idx_masks'].size(0),).to(inputs['part_idx_masks'].device)
        for part_idx in existing_parts:
            with torch.no_grad():
                part_mask = (inputs['part_idx_masks'] == part_idx ) * mask
                part_num += (part_mask.sum(-1) > 0).float()
                if not part_mask.any():
                    continue
                 
                part_rewards = (token_rewards * part_mask).sum(-1) / (part_mask.sum(-1) + 1e-10)
                try:
                    unique_rewards = torch.unique(token_rewards * part_mask)
                    assert unique_rewards.numel() <= 2   
                except:
                    print_rank_0(f"part {part_idx}, part_rewards: {part_rewards}")
                    print_rank_0(f"part_mask: {part_mask.tolist()}")
                    print_rank_0(f"token_rewards: {token_rewards.tolist()}")
                    print_rank_0(f"token_rewards * part_mask: {(token_rewards * part_mask).tolist()}")
                    print_rank_0(f"inputs['part_idx_masks']: {inputs['part_idx_masks'].tolist()}")
                part_kl_div = (kl_div * part_mask).sum(-1) 
                part_advantages = part_rewards - self.args.kl_coef * part_kl_div
            
            part_logprob = (logprobs * part_mask).sum(-1) / (part_mask.sum(-1) + 1e-10)
            old_logprob = (old_logprobs * part_mask).sum(-1) / (part_mask.sum(-1) + 1e-10)
            
            ratio_logits = torch.clamp(part_logprob-old_logprob, min=-10, max=10) 
            importance_ratio = ratio_logits.exp()
             
                 
            importance_ratio_clipped = torch.clip(importance_ratio, 1 - self.args.clip_range, 1 + self.args.clip_range)
            part_ppo_loss = - torch.minimum(part_advantages * importance_ratio, part_advantages * importance_ratio_clipped)
             
            total_loss += part_ppo_loss
             


        total_loss = total_loss / (part_num + 1e-10)
        print_rank_0(f"part_num: {part_num.tolist()}")
        print_rank_0(f"total_loss: {total_loss.tolist()}")


        weighted_loss = (total_loss * inputs['weights']).mean()  
        
        if sft_data is not None:
            model_outputs = model(
            input_ids=sft_data['input_ids'],
            attention_mask=sft_data['attention_mask'],
            labels=sft_data['labels']
        )
            logprobs, mask = compute_lm_loglikeli(model_outputs.logits, inputs['labels'])  
            train_sft_loss = - (logprobs * mask ).sum(-1) / mask.sum(-1)
            train_sft_loss = train_sft_loss.mean()
            
            weighted_loss = weighted_loss + 0.5 * train_sft_loss
            
            self.store_metrics({"train_sft_loss": train_sft_loss}, 'train')

        self.store_metrics({"total_loss": total_loss}, 'train')
         
        kl_div_avg = (kl_div.abs() * mask).sum(-1) / mask.sum(-1)
        sample_size = (1-inputs['sft_mask']).sum()
        kl_div_avg = (kl_div_avg * (1-inputs['sft_mask'])).sum() / sample_size if sample_size > 0 else 0
        self.store_metrics({"kl_div_avg": kl_div_avg}, 'train')
        
        importance_ratio = (importance_ratio * mask).sum(-1) / mask.sum(-1)
        importance_ratio = (importance_ratio * (1-inputs['sft_mask'])).sum() / sample_size if sample_size > 0 else 0
        self.store_metrics({"importance_ratio": importance_ratio}, 'train')

        

        if self.args.debug_mode:          
            print_rank_0(f"check loss : {total_loss}")
            print_rank_0(f"check weighted loss : {weighted_loss}")
             

        return (weighted_loss, model_outputs.logits) if return_outputs else weighted_loss



    def compute_loss_inkl_instancelevel(self, model, inputs, sft_data=None, return_outputs=False, use_sft_loss=False, stage="stage1"):


        if self.args.use_tf32_forward:
            with torch.autocast(dtype=torch.float32, device_type="cuda"):
                model_outputs = model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    labels=inputs['labels']
                )
        else:
            model_outputs = model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    labels=inputs['labels']
                )
        model_outputs.logits = model_outputs.logits.to(model.dtype)
         
         
        logprobs, mask = compute_lm_loglikeli_2(model_outputs.logits, inputs['labels'])  
        
        
         
        with torch.no_grad():
            self.ref_model.eval()
             
            if self.args.use_tf32_forward:
                with torch.autocast(dtype=torch.float32, device_type="cuda"):
                    ref_model_outputs = self.ref_model(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask']
                    )
            else:
                ref_model_outputs = self.ref_model(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask']
                    )
            ref_model_outputs.logits = ref_model_outputs.logits.to(model_outputs.logits.dtype)
            ref_logprobs, _ = compute_lm_loglikeli_2(ref_model_outputs.logits, inputs['labels'])
            old_logprobs, old_mask = compute_lm_loglikeli_2(inputs["old_forward_logits"], inputs['labels'])

            kl_div = old_logprobs - ref_logprobs  

            kl_div_sum = (kl_div * mask).sum(-1)
            
            advantages = inputs['instance_rewards'] - self.args.kl_coef * kl_div_sum
        

        logprob = (logprobs * mask).sum(-1) / (mask.sum(-1) + 1e-8)
        old_logprob = (old_logprobs * mask).sum(-1) / (mask.sum(-1)+ 1e-8)
         
        ratio_logits = torch.clamp(logprob - old_logprob, max=10, min=-10)   
        
        with torch.cuda.amp.autocast(enabled=False):
            importance_ratio = ratio_logits.float().exp().to(logprob.dtype)
            
            
        importance_ratio_clipped = torch.clip(importance_ratio, 1 - self.args.clip_range, 1 + self.args.clip_range)

        ppo_loss = - torch.minimum(advantages * importance_ratio, advantages * importance_ratio_clipped)


        weighted_loss = (ppo_loss * inputs['weights']).mean()  
        
        if sft_data is not None:
            model_outputs = model(
            input_ids=sft_data['input_ids'],
            attention_mask=sft_data['attention_mask'],
            labels=sft_data['labels']
        )
            logprobs, mask = compute_lm_loglikeli(model_outputs.logits, inputs['labels'])  
            train_sft_loss = - (logprobs * mask ).sum(-1) / mask.sum(-1)
            train_sft_loss = train_sft_loss.mean()
            
            weighted_loss = weighted_loss + 0.5 * train_sft_loss
            
            self.store_metrics({"train_sft_loss": train_sft_loss}, 'train')


        total_loss = weighted_loss.to(model.dtype)
            
        self.store_metrics({"total_loss": total_loss}, 'train')
         
        
        
        kl_div_avg = (kl_div.abs() * mask).sum(-1) / mask.sum(-1)
        sample_size = (1-inputs['sft_mask']).sum()
        kl_div_avg = (kl_div_avg * (1-inputs['sft_mask'])).sum() / sample_size if sample_size > 0 else 0
        self.store_metrics({"kl_div_avg": kl_div_avg}, 'train')
        
        importance_ratio = (importance_ratio * mask).sum(-1) / mask.sum(-1)
        importance_ratio = (importance_ratio * (1-inputs['sft_mask'])).sum() / sample_size if sample_size > 0 else 0
        self.store_metrics({"importance_ratio": importance_ratio}, 'train')

        

        if self.args.debug_mode:          
            print_rank_0(f"check loss : {total_loss}")
            print_rank_0(f"check weighted loss : {weighted_loss}")
             

        return (weighted_loss, model_outputs.logits) if return_outputs else weighted_loss



    @staticmethod
    def compute_kl_divergence(logprob, ref_logprob, kl_penalty='kl'):
        """
        if kl_penalty is not 'full'
        logprob: (batch_size, seq_len)
        ref_logprob: (batch_size, seq_len)
        if kl_penalty is 'full'
        logprob: (batch_size, seq_len, vocab_size)
        ref_logprob: (batch_size, seq_len, vocab_size)

        output: (batch_size, seq_len)
        """
        if kl_penalty == 'kl':
            return  logprob - ref_logprob
        
        if kl_penalty == 'abs':
            return torch.exp(logprob) * (logprob - ref_logprob).abs()
        
        if kl_penalty == 'mse':
            return 0.5 * torch.exp(logprob) * (logprob - ref_logprob).square()
        
        if kl_penalty == 'full':
            return nn.functional.kl_div(ref_logprob, logprob, log_target=True, reduction='none').sum(-1)


    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
         
         
        self.accelerator.free_memory()
        self._train_batch_size = self.args.train_batch_size
         
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                 
                if self.is_deepspeed_enabled:
                     
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

         
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=args.num_total_batches)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=args.num_total_batches)

        self.state = TrainerState(            
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]
            )
        
        
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

         
        self.state.global_step = 0
        self.state.episode = 0
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size
        self.state.max_steps = self.args.num_total_batches * self.args.num_mini_batches
        self.state.num_train_epochs = self.args.total_episodes / self.train_dataset_len
         
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(self.state.max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(self.state.max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(self.state.max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

         
        if args.gradient_checkpointing:
            if args.gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {}
            else:
                gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs

            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

         
         
         
        use_accelerator_prepare = True if model is self.model else False

         
        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=args.num_total_batches)

         
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                 
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )

        if self.is_fsdp_enabled:
            print_rank_0("FSDP is enabled, Setting self.model = self.model_wrapped = model")
            self.model = self.model_wrapped = model

         
        if model is not self.model:
            self.model_wrapped = model

         
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

         
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(self.model_wrapped, resume_from_checkpoint)
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

         
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.local_dataloader_batch_size*self.args.inner_step_num,
            shuffle=True,
             
            drop_last=True,   
        )
         
         
         
        



        def repeat_generator(data_loader, prepare=False):
            if prepare:
                while True:
                    yield from self.accelerator.prepare(data_loader)  
            else:
                while True:
                    yield from data_loader

         
        iter_dataloader = iter(repeat_generator(train_dataloader, prepare=True))

        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                 
                 
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:   
            max_steps = args.max_steps
             
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

         
        self.accelerator.print("***** Running training *****")
        self.accelerator.print(f"  Num examples = {self.num_examples(train_dataloader):,}")
        self.accelerator.print(f"  Num Epochs = {self.state.num_train_epochs:,}")
        self.accelerator.print(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            self.accelerator.print(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        self.accelerator.print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        self.accelerator.print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        self.accelerator.print(f"  Total optimization steps = {self.state.max_steps:,}")
         

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None
        
         
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            self.accelerator.print("  Continuing training from checkpoint, will skip to saved global_step")
            self.accelerator.print(f"  Continuing training from epoch {epochs_trained}")
            self.accelerator.print(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                self.accelerator.print(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )
        
         
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
             
             
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
         
         
         
         
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        self.accelerator.print("===training policy===")
        print_rank_0(f"self.accelerator.mixed_precision: {self.accelerator.mixed_precision}")
        self._globalstep_last_logged = self.state.global_step
        model.train()
        model.zero_grad()


         
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        print_rank_0(f"self.args.num_total_batches: {self.args.num_total_batches}")
        print_rank_0(f"self.args.batch_size: {self.args.batch_size}")
        total_batched_samples = 0
        last_batch_dataset = None
         
        all_rewards = []
        all_sample_num = 0
        all_margin_score = 0.
        current_best_score = 0. 
        best_veri_acc = 0.
        best_relative_alter_rate = 0.
        if self.args.restart_step > 0:
            step_with_optimizer = self.lr_scheduler.step_with_optimizer
            self.lr_scheduler.step_with_optimizer = False
            
            print_rank_0(f"skip {self.args.restart_step} step of lr_scheduler!")
            for _ in range(self.args.restart_step):
                _ = next(iter_dataloader)
                 

                AcceleratedScheduler.step(self.lr_scheduler)
            self.lr_scheduler.step_with_optimizer = step_with_optimizer
        
         
        use_prefix_baseline = True
        self.initialize_baseline(use_prefix=use_prefix_baseline)
        for update_id in range(self.args.restart_step+1, self.args.num_total_batches):
            
            self.check_for_nan(model)
            
             
            if args.use_instance_level:
                loss_func, loss_func_name = partial(self.compute_loss_inkl_instancelevel, stage="stage2"), "stage2"
            else:
                loss_func, loss_func_name = partial(self.compute_loss_splitpart_inkl_sentence, stage="stage2"), "stage2"
             
             
            
            self.state.episode += self.args.inner_step_num * self.args.batch_size
            batch_dataset = None
            all_relative_alter_rate = []
            all_veri_acc = []
            
            for _ in range(1):
                data = next(iter_dataloader)
                 
                data_batch_size = len(data['problem'])
                print_rank_0(f"data batch size: {len(data['problem'])}")
                generation_batch_size = 8
                start_time = time.time()
                with torch.no_grad():
                    with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                        with self.accelerator.autocast():
                            if args.use_instance_level:
                                use_bonus = False
                                use_veri_bonus = False
                                rloo_n = 4
                                batch_dataset_, all_rewards, all_sample_num, relative_alter_rate, none_sft_acc = self.generate_experience_incot_splitmask_rloo(unwrapped_model, data, all_rewards, all_sample_num, stage=loss_func_name, use_bonus=use_bonus, use_veri_bonus=use_veri_bonus, use_baseline=False, use_level_baseline=False, rloo_n=rloo_n)
                            else:
                                use_bonus = True
                                use_veri_bonus = True
                                rloo_n = 4
                                batch_dataset_, all_rewards, all_sample_num, relative_alter_rate, none_sft_acc = self.generate_experience_incot_splitmask(unwrapped_model, data, all_rewards, all_sample_num, stage=loss_func_name, use_bonus=use_bonus, use_veri_bonus=use_veri_bonus, use_baseline=False, use_level_baseline=False, use_prefix_baseline=use_prefix_baseline, rloo_n=rloo_n)
                            
                            data_batch_size *= rloo_n
                            if batch_dataset is None:
                                batch_dataset = batch_dataset_
                            else:
                                batch_dataset.add(batch_dataset_)
                             
                            all_veri_acc.append(none_sft_acc)
                        
                     
                    torch.cuda.empty_cache()
                    gc.collect()
                

                print_rank_0(f"Finish generation. Time usage: {time.time()-start_time}")

            
            print_rank_0(f"Saving checkpoint at Iter Num: {update_id}")
            self._save_checkpoint(model, trial=None, metrics=None)
            self.dump_baseline(use_prefix=use_prefix_baseline)

            mini_step = 0
            for ppo_epoch_idx in range(self.args.num_ppo_epochs):
                start_time = time.time()
                print_rank_0(f"args.per_device_train_batch_size: {args.per_device_train_batch_size}")
                
                data_batch_size = len(batch_dataset)
                print_rank_0(f"len(batch_dataset): {len(batch_dataset)}")
                min_dataloader = iter(repeat_generator(DataLoader(batch_dataset, batch_size=args.per_device_train_batch_size, shuffle=True)))
                 
                for _ in range(data_batch_size//args.per_device_train_batch_size):
                 
                    train_data = next(min_dataloader)
                    
                    mini_step += 1
                     
                     
                    with self.accelerator.accumulate(model):
                         
                        train_data = {k:v.to(self.accelerator.device) for k,v in train_data.items()}
                         
                        with self.accelerator.autocast():
                            loss = loss_func(model, train_data, use_sft_loss=False)
                            self.accelerator.backward(loss)
                         
                        self.accelerator.clip_grad_norm_(
                            parameters=model.parameters(),   
                            max_norm=1.0                    
                        )
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()
                        
                         
                    
                    if mini_step % args.local_mini_batch_size == 0:
                        self.state.global_step += 1
                        self.state.epoch = self.state.episode / self.train_dataset_len
                        self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                        logs: Dict[str, float] = {}
                        logs["learning_rate"] = self._get_learning_rate()
                        logs["loss_func"] = loss_func_name
                        self.log(logs)
                    else:
                        self.control = self.callback_handler.on_substep_end(args, self.state, self.control)
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                print_rank_0(f"Finish training step. Time usage: {time.time()-start_time}")
            

             
            if self.control.should_save:
                self._save_checkpoint(model, trial=None)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            torch.cuda.empty_cache()
            gc.collect()
            
         
         
        self.control = self.callback_handler.on_train_end(self.args, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(model, trial=None, metrics=None)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)


    def check_for_nan(self, model):
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                print_rank_0(f"========= Parameter {name} contains NaN values !!! ============")

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            if value != 0:
                self._stored_metrics[train_eval][key].append(value)


    def log(self, logs) -> None:
        train_eval = "train" 

        for key, metrics in self._stored_metrics[train_eval].items():
            try:
                logs[key] = torch.tensor(metrics).mean().item()
            except:
                print(f"{key}: {logs[key]}")
                 
         
        del self._stored_metrics[train_eval]
        return super().log(logs)