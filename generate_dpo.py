import os
import json
import torch
import argparse

from khaosz import Khaosz
from typing import List
from tqdm import tqdm

def batch_generate(
    queries: List[str],
    model: Khaosz,
    temperature: float, 
    top_k: int, 
    top_p: float,
    batch_size: int
) -> List:
    assert batch_size > 0
    sorted_queries = sorted(queries, key=lambda x: len(x), reverse=True)
    original_indices = {query: idx for idx, query in enumerate(queries)}
    
    responses = [None] * len(queries) 
    total_batches = (len(sorted_queries) + batch_size - 1) // batch_size 
    
    for i in tqdm(range(0, total_batches * batch_size, batch_size), desc="Generating responses"):
        batch_queries = sorted_queries[i: min(i + batch_size, len(queries))]
        if not isinstance(batch_queries, list):
            batch_queries = [batch_queries]
        
        batch_responses = model.batch_generate(
            queries=batch_queries,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        
        for batch_query, batch_response in zip(batch_queries, batch_responses):
            print((batch_query, batch_response))
        
        for query, response in zip(batch_queries, batch_responses):
            original_idx = original_indices[query] 
            responses[original_idx] = response  
            
    
    return responses

def dpo_generate(
    model: Khaosz,
    input_json_file: str,
    temperature: float,
    top_p: float,
    top_k: int,
    batch_size: int,
    question_key: str="question",
    recepted_key: str="recepted",
) -> List:  
    with open(input_json_file, "r") as f:
        json_file = json.load(f)
        item_size = len(json_file)
        queries = [item[question_key] for item in json_file]
        recepted = [item[recepted_key] for item in json_file]
    
    rejected = batch_generate(
        queries, 
        model,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        batch_size=batch_size
    )
    output_dict = []
    
    for i in range(item_size):
        output_dict.append({
            "question": queries[i],
            "recepted": recepted[i],
            "rejected": rejected[i]
        })

    return output_dict

def dpo_processor(
    model: Khaosz,
    input_json_file: str,
    output_json_file: str,
    batch_size: int,
    temperature: float,
    top_p: float,
    top_k: int,
    question_key: str="question",
    recepted_key: str="recepted",
):
    output_dict = dpo_generate(
        model=model, 
        input_json_file=input_json_file, 
        question_key=question_key, 
        recepted_key=recepted_key,
        batch_size=batch_size, 
        temperature=temperature,
        top_k=top_k,
        top_p=top_p
    )
    
    with open(output_json_file, "w", encoding='utf-8') as f:
        json.dump(output_dict, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    # 设置 argparse
    parser = argparse.ArgumentParser(description="Run DPO (Direct Preference Optimization) with a Khaosz model.")
    parser.add_argument("--input_json_file", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_json_file", type=str, required=True, help="Path to the output JSON file.")
    parser.add_argument("--question_key", type=str, default="question", help="Key for the question in the input JSON.")
    parser.add_argument("--recepted_key", type=str, default="recepted", help="Key for the accepted response in the input JSON.")
    parser.add_argument("--temperature", type=float, default=0.60, help="Temperature for generating responses.")
    parser.add_argument("--top_p", type=float, default=0.98, help="Top-p value for generating responses.")
    parser.add_argument("--top_k", type=int, default=20, help="Top-k value for generating responses.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generating responses.")

    args = parser.parse_args()

    # 加载模型
    model_dir = os.path.join(os.path.dirname(__file__), "params")
    model = Khaosz(model_dir)
    model = model.to(device='cuda', dtype=torch.float16)
    
    # 运行 DPO
    dpo_processor(
        model,
        input_json_file=args.input_json_file,
        output_json_file=args.output_json_file,
        question_key=args.question_key,
        recepted_key=args.recepted_key,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )