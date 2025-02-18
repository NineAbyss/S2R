import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import tqdm
import time
from threading import Lock  
import os
from multiprocessing import Value
from ctypes import c_int
import argparse


def load_problems(path):
    with open(path, 'r', encoding='utf-8') as f:
        problems = []
        for line in f:
            problems.append(json.loads(line))
        return problems

def create_prompt(problem_data):
    base_prompt = '''
You are a math teacher. I will give you a math problem and an answer, please tell me whether this answer is correct.
You cannot solve the problem step by step, you must use other methods to prove whether the answer is correct. 

* Question:
{question}

* Answer:
{answer}

* Your Prove:
'''
    return base_prompt.format(
        question=problem_data['problem'], 
        answer=problem_data['round_1_extracted_answer']
    )



def make_request_with_retry(data, max_retries=200):
    for attempt in range(max_retries):
        try:
            
            query_url = f'{args.api_url}'

            response = requests.post(
                query_url,
                json=data,
                headers=headers,
                timeout=30  
            )
            response.raise_for_status()  
            return response.json()
        except Exception as e:
            if attempt == max_retries - 1:  
                
                return None
            
            time.sleep(2)  

import random 
def load_reference_verifications(reference_files):
    reference_dict = {}
    for file_path in reference_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                
                for answer, verification in zip(data['round_1_extracted_answer'], data['verification']):
                    key = (data['problem'], answer)
                    if verification is not None:  
                        reference_dict[key] = verification
    return reference_dict

def process_single_problem(problem, file_path, lock, reference_dict, gpt_counter, reuse_counter, data_):
    problem['verification'] = []
    local_data = data_.copy()
    
    
    all_answers = problem['round_1_extracted_answer']
    answer_indices = {ans: [i for i, x in enumerate(all_answers) if x == ans] 
                     for ans in set(all_answers)}
    
    
    unique_answers = []
    
    if 'gold_extracted_answer' in problem and problem['gold_extracted_answer'] in answer_indices:
        unique_answers.append(problem['gold_extracted_answer'])
    
    
    for ans in answer_indices:
        if ans not in unique_answers and len(unique_answers) < 5:
            unique_answers.append(ans)
    
    
    kept_indices = []
    for ans in unique_answers:
        kept_indices.extend(answer_indices[ans])
    kept_indices.sort()
    
    
    problem['round_1_extracted_answer'] = [problem['round_1_extracted_answer'][i] for i in kept_indices]
    problem['round_1_response'] = [problem['round_1_response'][i] for i in kept_indices]
    for pred_answer in unique_answers:  
        
        key = (problem['problem'], pred_answer)
        if key in reference_dict:
            problem['verification'].append(reference_dict[key])
            with reuse_counter.get_lock():
                reuse_counter.value += 1

        else:
            prompt = create_prompt({
                'problem': problem['problem'],
                'round_1_extracted_answer': pred_answer
            })
            
            local_data["messages"] = [
            {"role": "user", "content": prompt}]
            response_ = make_request_with_retry(local_data)

            if response_ is None:
                return None
            with gpt_counter.get_lock():
                gpt_counter.value += 1
            problem['verification'].append(response_["choices"][0]["message"]["content"])

    
    original_verification = []
    for ans in problem['round_1_extracted_answer']:
        idx = unique_answers.index(ans)
        original_verification.append(problem['verification'][idx])
    problem['verification'] = original_verification
    with lock:
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(problem, ensure_ascii=False) + '\n')
    return True



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_file_path", type=str, default="DATA PATH")
    parser.add_argument("--output_file", type=str, default="OUTPUT PATH")
    parser.add_argument("--reference_file_path", type=str, default="REFERENCE FILE PATH")
    parser.add_argument("--api_key", type=str, default="API KEY")
    parser.add_argument("--api_url", type=str, default="API URL")
    parser.add_argument("--model_name_or_path", type=str, default="MODEL NAME")
    args = parser.parse_args()
    data_ = {'model': f'{args.model_name_or_path}',
        'messages': [],
        'n': 1,
        'temperature': 0.1,
        }

    headers = {
    "Authorization": f'Bearer {args.api_key}',
    "Content-Type": "application/json"
    }
    original_file_path = args.original_file_path
    output_file = args.output_file
    
    reference_files = [
        f'{args.reference_file_path}'
    ]
    reference_dict = load_reference_verifications(reference_files)
    print(f"Loaded {len(reference_dict)} reference verifications!")

    problems = load_problems(original_file_path)

    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            completed_lines = [json.loads(l) for l in f.readlines()]
        completed_unique_ids = set([l["unique_id"] for l in completed_lines])
        problems = [l for l in problems if l["unique_id"] not in completed_unique_ids]
        print(f"{len(completed_lines)} data already completed and filtered!")
    print(f"Start processing {len(problems)} data!")
     
    lock = Lock()  
    gpt_counter = Value(c_int, 0)  
    reuse_counter = Value(c_int, 0)  
    with ThreadPoolExecutor(max_workers=128) as pool:
        with tqdm.tqdm(total=len(problems)) as progress_bar:
            futures = [pool.submit(process_single_problem, problem.copy(), output_file, lock, reference_dict, gpt_counter, reuse_counter, data_)
                      for problem in problems]
            
            for future in as_completed(futures):
                is_valid = future.result()
                if is_valid:
                    progress_bar.update(1)
                    progress_bar.set_description(f"GPT-4 calls: {gpt_counter.value}, reuse: {reuse_counter.value}")

    print(f"Total GPT calls: {gpt_counter.value}, reuse: {reuse_counter.value}")
if __name__ == "__main__":
    main()

 
    