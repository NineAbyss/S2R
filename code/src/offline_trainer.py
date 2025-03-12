# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import itertools
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union, Literal
from copy import deepcopy
import deepspeed
import torch
import torch.nn as nn
from contextlib import contextmanager
import torch.nn.functional as F
# from accelerate import Accelerator
# from accelerate.utils import broadcast, gather_object
from datasets import Dataset
from transformers import (
    PreTrainedModel,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)

from arguments import OfflineRLConfig
from utils import print_object_on_main_process, print_rank_0

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
        # print("Unwrapping model for generation")
        with deepspeed.zero.GatheredParameters(model.parameters()):
            remove_hooks(model)
            yield model
            # yield accelerator.unwrap_model(model)
            add_hooks(model)
    else:
        yield unwrapped_model



def hooks_exist(engine):
    if engine.optimizer is not None and hasattr(engine.optimizer, "parameter_offload"):
        optimizer_offload = engine.optimizer.parameter_offload
    elif engine.optimizer is not None:
        optimizer_offload = engine.optimizer

    hooks = 0
    for hook in optimizer_offload.forward_hooks:
        hooks += 1
    if hooks > 0:
        return True
    return False

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
    
    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    
    # Enable model parallelism
    shift_logits = torch.clamp(shift_logits, min=-30, max=30)
    shift_labels = shift_labels.to(shift_logits.device)
    neg_logprobs = loss_fct(shift_logits, shift_labels).reshape(batch_size, -1) #  #[bs, seq_len]     # [bs * seq_len]
    ignore_mask = shift_labels != -100
    
    return -1* neg_logprobs, ignore_mask


class OfflineRLTrainer(Trainer):

    def __init__(
        self,
        config: OfflineRLConfig,
        processing_class,
        policy: nn.Module,
        ref_policy: nn.Module,
        train_dataset: Dataset,
        data_collator=None,
        eval_dataset=None,
        sft_data_module=None,
        # less commonly used
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
        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        if self._is_create_ref_model():
            self.ref_model = ref_policy
            for param in self.ref_model.parameters():
                param.requires_grad = False

            if self.is_deepspeed_enabled:
                self.ref_model, self.ref_model_engine = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        args = config
        self.train_dataset_len = len(train_dataset)
        # self.sft_data_module = sft_data_module
        #########
        # calculate various batch sizes
        #########
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

        #########
        # setup model, optimizer, and others
        #########
        # for module in [policy, ref_policy, reward_model]:
        #     disable_dropout_in_model(module)
        if args.stop_token and args.stop_token == "eos":
            args.stop_token_id = self.tokenizer.eos_token_id


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


    def compute_loss_process(
            self, model, inputs,
            return_outputs=False,
            num_items_in_batch=None, 
            sft_data=None, 
        ):

        model_outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            labels=inputs['labels']
        )
        logprobs, mask = compute_lm_loglikeli(model_outputs.logits, inputs['labels']) # logprobs: [bs, seq_len]
        
        # advantage
        with torch.no_grad():
            self.ref_model.eval()
            ref_model_outputs = self.ref_model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )

            ref_logprobs, _ = compute_lm_loglikeli(ref_model_outputs.logits, inputs['labels'])
            kl_div = self.compute_kl_divergence(logprobs, ref_logprobs, kl_penalty="kl")
            
            
        # token-level advantage
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)

        seq_len = min(
            inputs['token_rewards'].size(1) - 1,
            inputs['part_idx_masks'].size(1) - 1,
            mask.size(1) - 1,
            kl_div.size(1) - 1,
        )

        # align the sequence length
        token_rewards = inputs['token_rewards'][..., 1:seq_len + 1].contiguous()
        inputs['part_idx_masks'] = inputs['part_idx_masks'][..., 1:seq_len + 1].contiguous()
        mask = mask[..., 1:seq_len + 1].contiguous()
        kl_div = kl_div[..., 1:seq_len + 1].contiguous()
        logprobs = logprobs[..., 1:seq_len + 1].contiguous()
        ref_logprobs = ref_logprobs[..., 1:seq_len + 1].contiguous()
        total_loss = 0
        
        avg_reward = (token_rewards * mask).sum(-1) / mask.sum(-1)
        importance_ratios = []

        for part_idx in range(0, inputs['part_idx_masks'].max()+1):
            with torch.no_grad():
                part_mask = (inputs['part_idx_masks'] == part_idx) * mask
                if part_mask.sum() == 0:
                    continue
                part_rewards = (token_rewards * part_mask).sum(-1) / (part_mask.sum(-1) + 1e-10)
                part_kl_div = (kl_div * part_mask).sum(-1) / (part_mask.sum(-1) + 1e-10)
                part_advantages = part_rewards - self.args.lm_kl_coeff * part_kl_div
            
            part_logprob = (logprobs * part_mask).sum(-1) / (part_mask.sum(-1) + 1e-10)
            old_logprob = (ref_logprobs * part_mask).sum(-1) / (part_mask.sum(-1) + 1e-10)
            
            ratio_logits = torch.clamp(part_logprob-old_logprob, min=-10, max=10) 
            importance_ratio = (ratio_logits).exp()

            importance_ratio_clipped = torch.clip(importance_ratio, 1 - self.args.clip_range, 1 + self.args.clip_range)
            part_ppo_loss = - torch.minimum(part_advantages * importance_ratio, part_advantages * importance_ratio_clipped)
            total_loss += part_ppo_loss

            importance_ratios.append(importance_ratio)
        
        total_loss = total_loss / (part_idx + 1e-10)
        weighted_loss = (total_loss * inputs['weights']).mean()
        importance_ratio_avg = torch.stack(importance_ratios).mean()
        
        if sft_data is not None:
            model_outputs = model(
                input_ids=sft_data['input_ids'],
                attention_mask=sft_data['attention_mask'],
                labels=sft_data['labels']
            )
            logprobs, mask = compute_lm_loglikeli(model_outputs.logits, inputs['labels']) # logprobs: [bs, seq_len]
            train_sft_loss = - (logprobs * mask).sum(-1) / mask.sum(-1)
            train_sft_loss = train_sft_loss.mean()
            weighted_loss = weighted_loss + self.args.lm_sft_coeff * train_sft_loss
            self.store_metrics({"train_sft_loss": train_sft_loss}, 'train')

        if self.args.use_sft_loss and self.args.lm_sft_coeff:
            train_sft_loss = - (logprobs * mask).sum(-1) / mask.sum(-1)
            weighted_loss = weighted_loss + self.args.lm_sft_coeff * train_sft_loss
            self.store_metrics({"sft_loss": train_sft_loss}, 'train')
            self.store_metrics({"weight_loss": weighted_loss}, 'train')

        # log the metrics
        self.store_metrics({"avg_token_reward": avg_reward}, 'train')

        kl_div_avg = (kl_div.abs() * mask).sum(-1) / mask.sum(-1)
        sample_size = (1-inputs['sft_mask']).sum()
        kl_div_avg = (kl_div_avg * (1-inputs['sft_mask'])).sum() / sample_size if sample_size > 0 else 0
        self.store_metrics({"kl_div_avg": kl_div_avg}, 'train')
        
        importance_ratio = (importance_ratio * mask).sum(-1) / mask.sum(-1)
        importance_ratio = (importance_ratio * (1-inputs['sft_mask'])).sum() / sample_size if sample_size > 0 else 0

        self.store_metrics({"importance_ratio_avg": importance_ratio_avg}, 'train')

        if self.args.debug_mode:          
            print_rank_0(f"check loss : {total_loss}")
            print_rank_0(f"check weighted loss : {weighted_loss}")

        return (weighted_loss, model_outputs.logits) if return_outputs else weighted_loss


    def compute_loss(
            self, model, inputs,
            return_outputs=False,
            num_items_in_batch=None, 
            sft_data=None,
        ):
        if self.args.use_process_rl:
            return self.compute_loss_process(model, inputs, return_outputs,num_items_in_batch,  sft_data)
        else:
            return self.compute_loss_instancelevel(model, inputs, return_outputs, num_items_in_batch, sft_data)


    def compute_loss_instancelevel(self, model, inputs, return_outputs=False, num_items_in_batch=None,  sft_data=None):
        if self.args.debug_mode:
            print_rank_0(f"check inputs :{inputs}")
            
        model_outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            labels=inputs['labels']
        )

        with torch.no_grad():
            self.ref_model.eval()
            ref_model_outputs = self.ref_model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )

        logprobs, mask = compute_lm_loglikeli(model_outputs.logits, inputs['labels'])
        ref_logprobs, _ = compute_lm_loglikeli(ref_model_outputs.logits, inputs['labels'])
        
        logprob = (logprobs * mask).sum(-1) / (mask.sum(-1) + 1e-8)
        ref_logprob = (ref_logprobs * mask).sum(-1) / (mask.sum(-1) + 1e-8)
        
        # This is a sentence-level importance ratio
        importance_ratio = (logprob - ref_logprob).exp()
        importance_ratio_clipped = torch.clip(importance_ratio, 1 - self.args.clip_range, 1 + self.args.clip_range)

        kl_div = logprobs - ref_logprobs 
        kl_div_avg = (kl_div * mask).sum(-1) / mask.sum(-1)
        
        advantages = inputs['rewards'] - self.args.lm_kl_coeff * kl_div_avg

        ppo_loss = - torch.minimum(
            advantages * importance_ratio,
            advantages * importance_ratio_clipped
        )

        weighted_loss = (ppo_loss * inputs['weights']).mean() # [batch_size]

        if self.args.debug_mode:
            print_object_on_main_process("importance_ratio", importance_ratio)
            print_object_on_main_process("importance_ratio_clipped", importance_ratio_clipped)
            print_object_on_main_process("advantages", advantages)

        if sft_data is not None:
            sample_size, sft_size = (1-inputs['sft_mask']).sum(), (inputs['sft_mask']).sum()
            
            # logprob = (logprobs * mask).sum(-1) / mask.sum(-1)
            sft_loss = (- logprob * inputs['sft_mask']).sum() / sft_size if sft_size > 0 else sft_size
            ppo_loss = (ppo_loss * (1 - inputs['sft_mask'])).sum() / sample_size if sample_size > 0 else sample_size
            weighted_loss = weighted_loss + self.args.lm_sft_coeff * sft_loss + ppo_loss
            self.store_metrics({"sft_loss": sft_loss}, 'train')
        
        total_loss = weighted_loss.to(model.dtype)

        self.store_metrics({"ppo_loss": ppo_loss}, 'train')
        self.store_metrics({"kl_div": (kl_div*mask).sum()/mask.sum()}, 'train')
        
        pos_mask = (inputs['rewards'] > 0) * (1-inputs['sft_mask']) 
        neg_mask = (inputs['rewards'] < 0) * (1-inputs['sft_mask']) 
        kl_div_sentence = (kl_div * mask).sum(-1) / mask.sum(-1)

        sample_size = (1-inputs['sft_mask']).sum()
        kl_div_pos = (kl_div_sentence*pos_mask).sum()/pos_mask.sum() if pos_mask.sum() != 0 else 0
        kl_div_neg = (kl_div_sentence*neg_mask).sum()/neg_mask.sum() if neg_mask.sum() != 0 else 0

        importance_ratio = (importance_ratio * mask).sum(-1) / mask.sum(-1)
        importance_ratio = (importance_ratio*(1-inputs['sft_mask'])).sum()/sample_size.sum() if sample_size.sum() != 0 else 0
        
        self.store_metrics({"kl_div_pos": kl_div_pos}, 'train')
        self.store_metrics({"kl_div_neg": kl_div_neg}, 'train')
        self.store_metrics({"importance_ratio": importance_ratio}, 'train')

        if self.args.debug_mode:
            print_rank_0(f"check loss : {total_loss}")
            print_rank_0(f"check weighted loss : {weighted_loss}")

        return (weighted_loss, model_outputs.logits) if return_outputs else weighted_loss

    @staticmethod
    def compute_kl_divergence(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty='kl') -> torch.FloatTensor:
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


    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            if value != 0:
                self._stored_metrics[train_eval][key].append(value)


    def log(self, logs) -> None:
        train_eval = "train" 
        # if "loss" in logs else "eval"
        # print(f'train_eval: {train_eval} | self._stored_metrics[train_eval] {self._stored_metrics[train_eval]}')
        for key, metrics in self._stored_metrics[train_eval].items():
            try:
            # print(f'apply value: {key}: {metrics}')
                logs[key] = torch.tensor(metrics).mean().item()
            # print(f'Logs {logs} apply {logs[key]}')
            except:
            #     pass
                print(f"{key}: {logs[key]}")
                # print(metrics)
        # print(f'###################### log function######################\n_stored_metrics: {self._stored_metrics}\nlog{logs}')
        del self._stored_metrics[train_eval]
        return super().log(logs)