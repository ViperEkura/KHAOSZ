import os
import json
import torch
import warnings
import argparse

from module import Khaosz
from typing import List, Dict
from tqdm import tqdm

warnings.filterwarnings("ignore")

def batch_generate(
    queries,
    model: Khaosz,
    temperature=0.95, 
    top_k=50, 
    top_p=0.8 
) -> Dict:
    responses = []
    for query in tqdm(queries, desc="Generating responses"):
        response = model.generate(
            query=query, 
            temperature=temperature, 
            top_k=top_k, 
            top_p=top_p
        )
        responses.append(response)
    
    return responses

def dpo_generate(
    model: Khaosz,
    input_json_file: str,
    question_key: str="question",
    recepted_key: str="recepted",
) -> List:  
    with open(input_json_file, "r") as f:
        json_file = json.load(f)
        item_size = len(json_file)
        queries = [item[question_key] for item in json_file]
        recepted = [item[recepted_key] for item in json_file]
    
    rejected = batch_generate(queries, model)
    output_dict = []
    
    for i in range(item_size):
        output_dict.append({
            "question": queries[i],
            "recepted": recepted[i],
            "rejected": rejected[i]
        })

    return output_dict

def dpo(
    model: Khaosz,
    input_json_file: str,
    output_json_file: str,
    question_key: str="question",
    recepted_key: str="recepted",
):
    output_dict = dpo_generate(model, input_json_file, question_key, recepted_key)
    
    with open(output_json_file, "w") as f:
        json.dump(output_dict, f, indent=4)

if __name__ == "__main__":
    # 设置 argparse
    parser = argparse.ArgumentParser(description="Run DPO (Direct Preference Optimization) with a Khaosz model.")
    parser.add_argument("--input_json_file", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_json_file", type=str, required=True, help="Path to the output JSON file.")
    parser.add_argument("--question_key", type=str, default="question", help="Key for the question in the input JSON.")
    parser.add_argument("--recepted_key", type=str, default="recepted", help="Key for the accepted response in the input JSON.")
    args = parser.parse_args()

    # 加载模型
    model_dir = os.path.join(os.path.dirname(__file__), "params")
    model = Khaosz(model_dir)
    model = model.to(device='cuda', dtype=torch.float16)
    
    # 运行 DPO
    dpo(
        model,
        input_json_file=args.input_json_file,
        output_json_file=args.output_json_file,
        question_key=args.question_key,
        recepted_key=args.recepted_key
    )