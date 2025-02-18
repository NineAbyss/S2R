from src.rloo_trainer import RLOOTrainer
from src.rloo_utils import CustomDataset
import warnings
import transformers
from transformers import (
    HfArgumentParser,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    PreTrainedTokenizer
)
from arguments import RLOOConfig
import torch

from utils import print_rank_0, get_train_ds_config
import os
import wandb


def train():
    parser = HfArgumentParser((RLOOConfig,))
    args: RLOOConfig = parser.parse_args_into_dataclasses()[0]

    
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

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        model_max_length=args.max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )

    print_rank_0(f"Begin loading model from {args.model_path}...")
    print_rank_0(f"Loading model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, use_safetensors=True, torch_dtype=torch.bfloat16
    )

    
    if args.restart_step == 0:
        original_device = next(model.parameters()).device
        model = model.to('cpu')
        print(f"model device: {model.device}")
        tokenizer.add_special_tokens({"pad_token":"<pad>"})
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<pad>")
        tokenizer.pad_token = "<pad>"
    
        model.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer),mean_resizing=False) 
        model.config.vocab_size = len(tokenizer)
        model.get_input_embeddings().padding_idx = tokenizer.pad_token_id
        model.get_input_embeddings()._fill_padding_idx_with_zero()
        
        if "qwen2.5" in args.model_path.lower():
            tokenizer.end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
            tokenizer.end_token = "<|im_end|>"
            model.config.eos_token_id = tokenizer.end_token_id
            model.config.eos_token = tokenizer.end_token
        
        model = model.to(original_device)

    if args.ref_model_path is None:
        args.ref_model_path = args.model_path
        
    
    if torch.distributed.get_rank() == 0 and os.path.exists(args.dump_data_path):
        os.remove(args.dump_data_path)
    
    print_rank_0(f"Loading ref_model from {args.ref_model_path}...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.ref_model_path, trust_remote_code=True, use_safetensors=True, torch_dtype=torch.bfloat16
    )
    
    ref_model.config.pad_token_id = tokenizer.pad_token_id
    ref_model.resize_token_embeddings(len(tokenizer),mean_resizing=False) 
    ref_model.config.vocab_size = len(tokenizer)
    ref_model.get_input_embeddings().padding_idx = tokenizer.pad_token_id
    ref_model.get_input_embeddings()._fill_padding_idx_with_zero()

    print_rank_0(model)
    print_rank_0(f"Finish loading model from {args.model_path}.")

    model.is_parallelizable = True
    model.model_parallel = True


    train_data = args.rl_data_path
    dataset = CustomDataset(train_data)

    # args.report_to = ["wandb"]
    # os.environ["WANDB_PROJECT"] = "YOUR_WANDB_PROJECT" 
    # os.environ["WANDB_NAME"] = args.output_dir.split("/")[-1]
    # os.environ["WANDB_API_KEY"] = "YOUR_WANDB_API_KEY"
    os.environ["WANDB_MODE"] = "offline"
    
    
    args.deepspeed = get_train_ds_config(
            offload=True,
            dtype="bf16",
            stage=3,
            enable_hybrid_engine=True,
            release_inference_cache=True,
            pin_parameters=True, 
            max_out_tokens=args.max_length,
            )
    
    
    trainer = RLOOTrainer(
        config=args,
        processing_class=tokenizer,
        policy=model,
        ref_policy=ref_model,
        train_dataset=dataset,
        sft_data_module=None,
    )
    with torch.autocast("cuda"):
        trainer.train()

if __name__ == "__main__":
    train()