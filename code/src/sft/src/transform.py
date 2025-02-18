from transformers import Qwen2TokenizerFast
from argparse import ArgumentParser
from utils import read_json_or_jsonl_data
import json
from tqdm import tqdm
from random import shuffle


def get_parser():
    parser = ArgumentParser()
    parser.add_argument(
        '--model_type',
        type=str
    )
    parser.add_argument(
        '--data_path',
        type=str
    )
    parser.add_argument(
        '--model_name_or_path',
        type=str
    )

    parser.add_argument(
        '--split',
        type=float,
        default=None
    )
    parser.add_argument(
        '--save_dir',
        type=str
    )
    parser.add_argument(
        '--save_name'
    )
    return parser


def main(args):
    dataset = read_json_or_jsonl_data(args.data_path)

    if args.model_type == 'qwen':
        tokenizer = Qwen2TokenizerFast.from_pretrained(args.model_name_or_path)
    
    new_dataset = []
    for data in tqdm(dataset):
        messages = [{"role": "user", "content": data['prompt']},
                    {"role": "assistant", "content": data['answer']}]
        prompt = tokenizer.apply_chat_template(messages[:1], add_generation_prompt=True, tokenize=False)
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        answer = text[len(prompt):]
        new_dataset.append({
                "prompt": prompt,
                "answer": answer,
            })
    

    if args.split is None:
        with open(f"{args.save_dir}/{args.save_name}.json", 'w') as f:
            json.dump(new_dataset, f, ensure_ascii=False, indent=2)
    else:
        shuffle(new_dataset)
        with open(f"{args.save_dir}/{args.save_name}_train.json", 'w') as f:
            json.dump(new_dataset[:int(args.split*len(new_dataset))], f, ensure_ascii=False, indent=2)
        with open(f"{args.save_dir}/{args.save_name}_test.json", 'w') as f:
            json.dump(new_dataset[int(args.split*len(new_dataset)):], f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)