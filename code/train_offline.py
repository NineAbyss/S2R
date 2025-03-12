import os
import torch
from typing import Dict
from arguments import OfflineRLConfig
from src.offline_trainer import OfflineRLTrainer
from transformers import (
    HfArgumentParser,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer
)
from data_process_offline_rl import OfflineRLDataset, DataCollatorForOfflineRL
from utils import print_rank_0, get_train_ds_config


def make_sft_data_module(tokenizer: PreTrainedTokenizer, args) -> Dict:
    """Make dataset and collator for PPO fine-tuning."""

    train_set = OfflineRLDataset(
        args.train_data_path,
        tokenizer=tokenizer, 
        format_mode=args.format_mode
    )

    if args.debug_mode:
        print_rank_0(">>> check text dataset:")
        print_rank_0(f">>> {train_set[0]}")

    dataset_dict = {
        "train_dataset": train_set,
        "eval_dataset": None,
        "data_collator": DataCollatorForOfflineRL(tokenizer=tokenizer),
    }
    return dataset_dict


def train():
    parser = HfArgumentParser((OfflineRLConfig,))
    args: OfflineRLConfig = parser.parse_args_into_dataclasses()[0]

    # set restart
    if "restart" in args.output_dir and args.restart_step == 0:
        try:
            restart_step = eval(args.model_path.split("checkpoint-")[-1])
            if isinstance(restart_step, int):
                args.restart_step = restart_step
        except:
            pass

    if args.restart_step > 0:
        print(f"Restart from step {args.restart_step}")
        

    print_rank_0(f"Setting tokenizer from {args.model_path}...")
    if "deepseek" in args.model_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            model_max_length=args.max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            model_max_length=args.max_length,
            padding_side="right",
            use_fast=False,
            trust_remote_code=True,
        )

    # for bug of llama3
    if "llama" in args.model_path.split("/")[-1].lower() and tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        tokenizer.eos_token =  "<|eot_id|>"

    print_rank_0(f"Begin loading model from {args.model_path}...")    
    print_rank_0(f"Loading model from {args.model_path}...")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, use_safetensors=True
    )
    model = model.bfloat16()

    if args.restart_step == 0:
        tokenizer.add_special_tokens({"pad_token":"<pad>"})
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<pad>")
        tokenizer.pad_token = "<pad>"
    
        model.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))
        model.get_input_embeddings().padding_idx = tokenizer.pad_token_id
        model.get_input_embeddings()._fill_padding_idx_with_zero()

    if args.ref_model_path is None:
        args.ref_model_path = args.model_path
        
    if args.step_num_per_stage == 0:
        args.step_num_per_stage = args.save_steps
    
    if torch.distributed.get_rank() == 0 and os.path.exists(args.dump_data_path):
        os.remove(args.dump_data_path)
    
    print_rank_0(f"Loading ref_model from {args.ref_model_path}...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.ref_model_path, trust_remote_code=True, use_safetensors=True
    )
    ref_model.config.pad_token_id = tokenizer.pad_token_id
    ref_model.resize_token_embeddings(len(tokenizer))
    ref_model.get_input_embeddings().padding_idx = tokenizer.pad_token_id
    ref_model.get_input_embeddings()._fill_padding_idx_with_zero()

    print_rank_0(model)
    print_rank_0(f"Finish loading model from {args.model_path}.")

    model.is_parallelizable = True
    model.model_parallel = True

    args.report_to = ["wandb"]
    os.environ["WANDB_PROJECT"] = "train_offline_rl" 
    os.environ["WANDB_NAME"] = args.output_dir.split("/")[-1]
    os.environ["WANDB_API_KEY"] = "xxxxxxxxxx"
    os.environ["WANDB_MODE"] = "offline"

    train_dataset = OfflineRLDataset(
        args.train_data_path,
        tokenizer=tokenizer,
        format_mode=args.format_mode,
        reward_load_path=args.reward_load_path,
        reward_save_patn=args.reward_save_path,
        debug_mode=args.debug_mode,
        use_bonus=args.use_bonus,
        use_veri_bonus=args.use_veri_bonus,
        use_prefix_baseline=args.use_prefix_baseline,
        use_position_baseline=args.use_position_baseline,
        use_level_baseline=args.use_level_baseline,
        use_accuracy_baseline=args.use_accuracy_baseline,
        use_softmax_norm=args.use_softmax_norm,
        reward_delay_factor=args.reward_delay_factor,
        use_process_rl=args.use_process_rl
    )
    
    args.deepspeed = get_train_ds_config(
        offload=True,
        dtype="bf16",
        stage=3,
        enable_hybrid_engine=True,
        release_inference_cache=True,
        pin_parameters=True,
        max_out_tokens=args.max_length,
    )
    
    trainer = OfflineRLTrainer(
        config=args,
        processing_class=tokenizer,
        policy=model,
        ref_policy=ref_model,
        train_dataset=train_dataset,
        data_collator=DataCollatorForOfflineRL(tokenizer=tokenizer)
    )
    with torch.autocast("cuda"):
        trainer.train()
    
    trainer.save_model()

if __name__ == "__main__":
    train()
