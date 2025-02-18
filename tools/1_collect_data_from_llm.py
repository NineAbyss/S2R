import os
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import argparse
import os
import json
import re
import requests
from answer_extraction import extract_answer
from transformers import AutoTokenizer


HEADERS = {"Content-Type": "application/json"}

AVAILABLE_URLs = [
"0.0.0.0"
]




def generate_two_step_response(problem):
    
    round_1_instruction = problem


    data = {
        "prompt": None,
        "top_p": 1.0,
        "temperature": 0.7,
        "max_tokens": 2048,
        "stop_token_ids": [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
        "n": 5
        }
    
    query_url = f"http://{random.choice(AVAILABLE_URLs)}:8081/generate"

    messages = [
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        { "role": "user","content": round_1_instruction},
    ]
    prompt_1 = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    
    data["prompt"] = prompt_1
    response = eval(requests.post(query_url, json = data).text)['text']
    # print(response) 
    if response[0].endswith("<|eot_id|>"):
        response[0] = response[0][:-len("<|eot_id|>")]
    

    
    result_dict = {
        "problem": problem,
        "round_1_instruction": round_1_instruction,
        "prompt_1": prompt_1,
        "round_1_response": response,
        "url": query_url
    }
    
    return result_dict
        

def process_single_line(data_line, output_file):

    result_dict = generate_two_step_response(data_line["problem"])
    result_dict["subject"] = data_line.get("subject", None)
    result_dict["level"] = data_line.get("level", None)
    result_dict["solution"] = data_line["solution"]
    result_dict["unique_id"] = data_line["unique_id"]
    
    round_1_extracted_answer = [extract_answer(r) for r in result_dict["round_1_response"]] 
    if data_line.get("gold_extracted_answer", None) is not None:
        gold_extracted_answer = data_line["gold_extracted_answer"]
    else: 
        gold_extracted_answer = extract_answer(result_dict["solution"])
    result_dict["round_1_extracted_answer"] = round_1_extracted_answer
    result_dict["gold_extracted_answer"] = gold_extracted_answer
    
    with open(output_file, "a") as f:
        f.write(json.dumps(result_dict, ensure_ascii=False)+"\n")
    
    return True



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="PATH TO YOUR BASE MODEL")
    parser.add_argument("--data_dir", type=str, default="DATA PATH")
    parser.add_argument("--data_name", type=str, default="DATA NAME")
    parser.add_argument("--output_file", type=str, default="OUTPUT PATH")
    args = parser.parse_args()

    data_dir = args.data_dir
    data_name = args.data_name

    output_file = args.output_file+'/output.jsonl'
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    with open(os.path.join(data_dir, data_name), "r") as f:
        lines = [json.loads(l) for l in f.readlines()]
    
    print(f"{len(lines)} data loaded from {os.path.join(data_dir, data_name)}!")
    print(f"Dumping to {output_file}")
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            completed_lines = [json.loads(l) for l in f.readlines()]
        completed_unique_ids = set([l["unique_id"] for l in completed_lines])
        lines = [l for l in lines if l["unique_id"] not in completed_unique_ids]
        print(f"{len(completed_lines)} data already completed and filtered!")

    print(f"Start processing {len(lines)} data!")
    with ThreadPoolExecutor(max_workers=128) as pool:
        with tqdm.tqdm(total=len(lines)) as progress_bar:

            futures = [pool.submit(process_single_line, data_line=line, output_file=output_file)
                       for line in lines]
            
            for future in as_completed(futures):
                is_valid = future.result()
                if is_valid:
                    progress_bar.update(1)
    
