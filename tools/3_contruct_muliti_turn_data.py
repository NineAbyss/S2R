import json
import random
import os
from answer_extraction import extract_answer, extract_boxed_answers
import requests
import tqdm
import time
from transformers import AutoTokenizer
from answer_extraction import answer_corrected_match
import collections
import argparse
def get_soft_answer_correction(gold_answer, output_answer):
    if "=" in gold_answer:
        gold_answer = gold_answer.strip().split("=")[-1].strip()
    if "=" in output_answer:
        output_answer = output_answer.strip().split("=")[-1].strip()
    if gold_answer == output_answer:
        return True
    if answer_corrected_match(gold_answer, output_answer) or answer_corrected_match(output_answer, gold_answer):
        
        return True
    
    return False

def is_incorrect_verification(sentence):
    incorrect_patterns = [
        "is incorrect",
        "is likely incorrect",
        "is unlikely correct",
        "answer is wrong",
        "incorrect answer",
        "not the correct answer",
        "answer is not correct",
        "solution is incorrect",
        "calculation is wrong",
        "result is incorrect"   
    ]
    return any(pattern in sentence.lower() for pattern in incorrect_patterns)

def is_correct_verification(sentence):
    correct_patterns = [
        "is correct",
        "appears to be correct", 
        "answer is reasonable",
        "solution is correct",
        "calculation is correct",
        "result is correct",
        "correctly solved",
        "answer checks out"
    ]
    return any(pattern in sentence.lower() for pattern in correct_patterns)

failed_requests_count = 0




def process_lines(lines, reference_answers, response_verification_dict):


    processed_results = []
    for line in tqdm.tqdm(lines):
        
        result = {
            "unique_id": line['unique_id'],
            "round_1_instruction": line['round_1_instruction'],
            "problem": line['problem'],
            "correct_response_veri": None,
            "incorrect_response_veri": None,
            "round_1_extracted_answer": line['round_1_extracted_answer'],
            "gold_extracted_answer": line['gold_extracted_answer'],
            
        }
        
        unique_id = line['unique_id']
        gold_extracted_answer = line['gold_extracted_answer']
        
        
        mistral_veri_list = response_verification_dict.get(unique_id, {}).get("mistral", [])
        qwen1_veri_list = response_verification_dict.get(unique_id, {}).get("qwen1", [])
        qwen2_veri_list = response_verification_dict.get(unique_id, {}).get("qwen2", [])
        res_veri_list = mistral_veri_list + qwen1_veri_list + qwen2_veri_list
        reusable_verifications = []
        for pair in res_veri_list:
            resp = pair["response"]
            verifications = pair["verification"]
            other_boxed_answer = extract_boxed_answers(resp)
            
            for r1_resp in line['round_1_response']:
                if other_boxed_answer == extract_boxed_answers(r1_resp):
                    reusable_verifications.append({
                        "response": r1_resp,
                        "verification": verifications
                    })
                    break  


        res_veri_list = reusable_verifications
         
        if not res_veri_list:
            continue
        
        
        correct_response_veri_prior = []  
        correct_response_veri_later = []  
        incorrect_response_veri_prior = []  
        incorrect_response_veri_later = []  
        
        exist_answer_set = set()
        for res_veri in res_veri_list:
            response = res_veri["response"]
            verification = res_veri["verification"]
            
            extracted_answer = extract_boxed_answers(response)
            
            if not extracted_answer:
                continue
            else:
                extracted_answer = extracted_answer[-1]
                extracted_answer = extracted_answer.split("=")[-1].strip()

            
            is_answer_correct = get_soft_answer_correction(gold_extracted_answer, extracted_answer)
            veri_answer = "" 
            sentences = verification.lower().strip().split('.')
            last_sentence = sentences[-2] if verification.endswith('.') else sentences[-1]
            
       
            if is_incorrect_verification(last_sentence):
                veri_answer = "incorrect"
            elif is_correct_verification(last_sentence):
                veri_answer = "correct"
            elif "plausible" in last_sentence.lower():
                veri_answer = "correct"
                verification = verification.replace("is plausible","is correct")
            elif "cannot" in last_sentence.lower() and "verif" in last_sentence.lower():
                continue
            elif len(sentences) >= 3:
                second_last_sentence = sentences[-3] if verification.endswith('.') else sentences[-2]
                if is_incorrect_verification(second_last_sentence):
                    veri_answer = "incorrect"
                    verification = '.'.join(verification.split('.')[:-1]) + '.'  
                elif is_correct_verification(second_last_sentence):
                    veri_answer = "correct"
                    verification = '.'.join(verification.split('.')[:-1]) + '.'  
            else:
                
                continue
            if is_answer_correct and veri_answer == "correct":
                if res_veri in qwen1_veri_list:
                    correct_response_veri_prior.append({"response": response, "verification": verification})
                else:
                    correct_response_veri_later.append({"response": response, "verification": verification})
            
            
            elif not is_answer_correct and veri_answer == "incorrect":
                if extracted_answer in exist_answer_set:
                    continue
                
                if res_veri in qwen1_veri_list:
                    incorrect_response_veri_prior.append({"response": response, "verification": verification})
                else:
                    incorrect_response_veri_later.append({"response": response, "verification": verification})
                exist_answer_set.add(extracted_answer)
                
        if correct_response_veri_prior:
            result["correct_response_veri"] = correct_response_veri_prior
        else:
            result["correct_response_veri"] = correct_response_veri_later
        
        result["incorrect_response_veri_prior"] = incorrect_response_veri_prior
        result["incorrect_response_veri_later"] = incorrect_response_veri_later
        processed_results.append(result)
    
    return processed_results

def construct_answer(res_veri_list):
    '''
    res_veri_list: [{"response": ,"verification": }]
    '''
    answer = ""
    for idx, res_veri in enumerate(res_veri_list):
        answer += f'{res_veri["response"]}\n\nWait, let me recheck my solution.\n\n{res_veri["verification"]}'
        if idx != len(res_veri_list) - 1:
            answer += "\n\nLet me try again.\n\n"
    return answer



def select_train_data(processed_results):
    '''
    processed_results: {"unique_id": ,"problem": ,"correct_response_veri": [{"response": ,"verification": }], "incorrect_response_veri": [{"response": ,"verification": }]}
    '''
    one_res_data = []
    multi_res_data = []
    train_data = []
    data_length_count = collections.defaultdict(int)
    for result in processed_results:
        unique_id = result["unique_id"]
        problem = result["problem"]
        round_1_instruction = problem

       
        correct_response_veri = result["correct_response_veri"]
        incorrect_response_veri = result["incorrect_response_veri_prior"] + result["incorrect_response_veri_later"]
        
        if len(correct_response_veri) == 0:
            
            continue

        wrong_count = len(result.get("round_1_extracted_answer", []))  
        correct_count = sum(1 for ans in result.get("round_1_extracted_answer", []) 
                          if ans == result.get("gold_extracted_answer"))  
        wrong_count = wrong_count - correct_count  
        
        
        if 0 <= wrong_count <= 0:
            
            res_veri = random.choice(correct_response_veri)
            if random.random() < 0.76*0.45*0.7:
                train_data.append({
                    "prompt": round_1_instruction,
                    "answer": construct_answer([res_veri])
                })
                data_length_count[1] += 1
            
        elif 1<= wrong_count <= 1 and len(incorrect_response_veri) >= 1:
            
                response_veri = random.sample(incorrect_response_veri, 1) + random.sample(correct_response_veri, 1)
                if random.random() <= 0.5:
                    train_data.append({
                        "prompt": round_1_instruction,
                        "answer": construct_answer(response_veri)
                    })
                    data_length_count[2] += 1

            
        elif 2 <= wrong_count <= 3:
            if len(incorrect_response_veri) >= 2:
            
                response_veri = random.sample(incorrect_response_veri, 2) + random.sample(correct_response_veri, 1)
                if random.random() <= 0.5:
                    train_data.append({
                        "prompt": round_1_instruction,
                        "answer": construct_answer(response_veri)
                    })
                    data_length_count[3] += 1
            elif len(incorrect_response_veri) == 1:
                response_veri = random.sample(incorrect_response_veri, 1) + random.sample(correct_response_veri, 1)
                if random.random() <= 1:
                    train_data.append({
                        "prompt": round_1_instruction,
                        "answer": construct_answer(response_veri)
                    })
                    data_length_count[2] += 1
        elif 4<=wrong_count <= 20:
            if len(incorrect_response_veri) >= 3:
                
                
                response_veri = random.sample(incorrect_response_veri, 3) + random.sample(correct_response_veri, 1)
                if random.random() <= 0.5:
                    train_data.append({
                        "prompt": round_1_instruction,
                        "answer": construct_answer(response_veri)
                    })
                    data_length_count[4] += 1
            elif len(incorrect_response_veri) == 2:
                response_veri = random.sample(incorrect_response_veri, 2) + random.sample(correct_response_veri, 1)
                if random.random() <= 1:
                    train_data.append({
                        "prompt": round_1_instruction,
                        "answer": construct_answer(response_veri)
                    })
                    data_length_count[3] += 1
            elif len(incorrect_response_veri) == 1:
                response_veri = random.sample(incorrect_response_veri, 1) + random.sample(correct_response_veri, 1)
                if random.random() <= 1:
                    train_data.append({
                        "prompt": round_1_instruction,
                        "answer": construct_answer(response_veri)
                    })
                    data_length_count[2] += 1
    return train_data, data_length_count


def split_data(res_data, tokenizer):
    import re
    
    new_data = []
    for data in tqdm.tqdm(res_data):
        prompt = data["prompt"]
        response_text = data["answer"]
        split_texts = re.split("(Wait,|Let me try again.\n\n)", response_text)

        initial_answer = split_texts[0]
        
        
        if len(split_texts) == 3:
            new_data.append(data)
            continue
        
        else:
            prompt += initial_answer
            
            last_split_token = None
            for idx, split in enumerate(split_texts[1:]):
                if split == "Let me try again.\n\n" or split == "Wait,":
                    last_split_token = split
                    continue
                    
                else:
                    
                    if last_split_token == "Wait,":
                        
                        prompt_delete_last = tokenizer.decode(tokenizer(prompt)["input_ids"][:-1], skip_special_tokens=False)

                        
                        answer = prompt[len(prompt_delete_last):] + last_split_token + split 
                        if idx != len(split_texts[1:]) - 1:
                            answer += "Let me try again.\n\n"
                        new_data.append({
                            "prompt": prompt_delete_last,
                            "answer": answer
                        })
                        prompt += last_split_token + split
                    
                    else:
                        if idx == len(split_texts[1:]) - 3:
                            prompt_delete_last = tokenizer.decode(tokenizer(prompt)["input_ids"][:-1], skip_special_tokens=False)
                            
                            answer = prompt[len(prompt_delete_last):] + last_split_token + split + "Wait, let me recheck my solution.\n\n"
                            new_data.append({
                                "prompt": prompt_delete_last,
                                "answer": answer
                            })
                            prompt += last_split_token + split
                        else:
                            prompt += last_split_token + split
                
                        
            assert len(prompt) == len(data["prompt"]) + len(data["answer"])
            
    return new_data


def filter_veri_data(veri, response):
    
    veri_len = len(veri.split("\n"))
    response_len = len(response.split("\n"))
    
    if veri_len > response_len:
        return False
    
    if len(veri) > len(response):
        return False
    
    if not veri.strip().startswith("Use"):
        return False
    
    if "Use direct" in veri:
        return False
    
    return True

    
def transform_data(dataset):

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    new_dataset = []
    for data in tqdm.tqdm(dataset):
        messages = [{"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                    {"role": "user", "content": data['prompt']},
                    {"role": "assistant", "content": data['answer']}]
        prompt = tokenizer.apply_chat_template(messages[:2], add_generation_prompt=True, tokenize=False)
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        answer = text[len(prompt):]
        new_dataset.append({
                "prompt": prompt,
                "answer": answer,
            })
    
    return new_dataset, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--response_file_path", type=str, default="RESPONSE FILE PATH")
    parser.add_argument("--output_file", type=str, default="OUTPUT PATH")
    parser.add_argument("--model_name_or_path", type=str, default="MODEL NAME")
    parser.add_argument("--verification_file_path", type=str, default="VERIFICATION FILE PATH")
    args = parser.parse_args()
    file_list = [
        args.response_file_path
    ]

    for file_path in file_list:
        with open(file_path, "r") as f:
            lines = [json.loads(l) for l in f.readlines()] 
    all_results = []
    
    
    reference_answers = None

    response_veri_dict = {}
    
    def get_answer_veri(file_path, key_name):
        
        with open(file_path, "r") as f:
            lines = [json.loads(l) for l in f.readlines()]
        for line in lines:
            if line['unique_id'] not in response_veri_dict:
                response_veri_dict[line['unique_id']] = {}
            response_veri_dict[line['unique_id']][key_name] = []  
            for idx in range(len(line["round_1_response"])):
                
                if not line["round_1_response"][idx] or not line["verification"][idx]:
                    continue
                
                response_veri_dict[line['unique_id']][key_name].append({
                    "response": line["round_1_response"][idx],
                    "verification": line["verification"][idx].replace("original answer", "answer").replace(" without solving the problem step by step", "")
                })

    get_answer_veri(args.verification_file_path, key_name="qwen1")
    
    
    
    
 
    
    processed_results = process_lines(lines, reference_answers, response_veri_dict)
    
    
    
    res_data, data_length_count = select_train_data(processed_results)
        
    print(data_length_count)

        
    res_data, tokenizer = transform_data(res_data)
    raw_output_path = args.output_file
    with open(raw_output_path, "w") as f:
        json.dump(res_data, f, indent=4, ensure_ascii=False)
    breakpoint()

    res_data = split_data(res_data, tokenizer)
    
    output_path = args.output_file
    with open(output_path, "w") as f:
        json.dump(res_data, f, indent=4, ensure_ascii=False)