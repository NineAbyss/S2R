"""Code for evaluation of model."""

import argparse
import json
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from utils import jload, print_rank_0

IGNORE_INDEX = -100

# Baichuan format
PROMPT_FORMAT_SINGLE_BC = "<reserved_106> {instruction}<reserved_107> "

# Qwen format
PROMPT_FORMAT_SINGLE_QWEN = (
    "\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
)

# InternLM format
PROMPT_FORMAT_SINGLE_IT = (
    "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
)


def run_planning_test(test_input, model, tokenizer, out_path, model_type):
    """Run planning test on single input."""
    generation_config = GenerationConfig(
        temperature=0.3,
        do_sample=True,
        max_new_tokens=256,
        top_k=10,
        top_p=0.85,
        repetition_penalty=1.05,
        eos_token_id=tokenizer.eos_token_id,
    )

    output_list = []
    with open(out_path, "w", encoding="utf-8") as outfile:
        for i in tqdm(range(len(test_input))):
            cur_test_input = test_input[i]["instruction"]

            if model_type == "Baichuan_token":
                cur_test_input = PROMPT_FORMAT_SINGLE_BC.format(
                    instruction=cur_test_input
                )
            elif model_type == "Qwen_token":
                cur_test_input = PROMPT_FORMAT_SINGLE_QWEN.format(
                    instruction=cur_test_input
                )
            elif model_type == "Intern_token":
                cur_test_input = PROMPT_FORMAT_SINGLE_IT.format(
                    instruction=cur_test_input
                )
            else:
                raise ValueError(f"Invalid model type: {model_type}")

            input_tokens = tokenizer(cur_test_input, return_tensors="pt")
            input_tokens = input_tokens.to("cuda")
            print(input_tokens)

            with torch.no_grad():
                generation_output = model.generate(
                    **input_tokens, generation_config=generation_config
                )
                generation_output = tokenizer.decode(generation_output[0])

                output_list.append(generation_output)

            tmp = {}
            tmp["instruction"] = test_input[i]["instruction"]
            tmp["context"] = test_input[i]["context"]
            tmp["orig_planning"] = test_input[i]["output"]
            tmp["model_planning"] = generation_output

            outfile.write(json.dumps(tmp, ensure_ascii=False) + "\n")
            outfile.flush()
            os.fsync(outfile.fileno())


def eval_planning():
    """Main function to evaluate planning on a test dataset."""
    parser = argparse.ArgumentParser(description="Parameters")
    parser.add_argument("--data_path", default="", type=str, help="config path")
    parser.add_argument("--model_path", default="", type=str, help="model path")
    parser.add_argument("--output_path", default="", type=str, help="output path")
    parser.add_argument("--model_type", default="", type=str, help="model type")
    args = parser.parse_args()
    print_rank_0(args)

    # Setup tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, use_fast=False, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, device_map="auto", trust_remote_code=True
    )

    model.eval()

    test_input = jload(args.data_path)

    run_planning_test(test_input, model, tokenizer, args.output_path, args.model_type)


if __name__ == "__main__":
    eval_planning()
