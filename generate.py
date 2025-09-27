import os
import torch
import json
import torch
import argparse

from khaosz import Khaosz
from typing import List
from tqdm import tqdm


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def batch_generate(
    model: Khaosz,
    queries: List[str],
    temperature: float, 
    top_k: int, 
    top_p: float,
    batch_size: int,
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
            print(f"Q: {batch_query[:50]} \nR: {batch_response[:50]})")
        
        for query, response in zip(batch_queries, batch_responses):
            original_idx = original_indices[query] 
            responses[original_idx] = response  

    return responses


def processor(
    model: Khaosz,
    input_json_file: str,
    output_json_file: str,
    batch_size: int,
    temperature: float,
    top_p: float,
    top_k: int,
    question_key: str="question",
):
    with open(input_json_file, "r", encoding='utf-8') as f:
        input_dict = [json.loads(line) for line in f]
        queries = [item[question_key] for item in input_dict]
    
    output_dict = batch_generate(
        model=model,
        queries=queries,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        batch_size=batch_size
    )
    
    with open(output_json_file, "w", encoding='utf-8') as f:
        json.dump(output_dict, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run generate with a Khaosz model.")
    
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the model directory.")
    parser.add_argument("--input_json_file", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output_json_file", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--question_key", type=str, default="question", help="Key for the question in the input JSON.")
    parser.add_argument("--temperature", type=float, default=0.60, help="Temperature for generating responses.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p value for generating responses.")
    parser.add_argument("--top_k", type=int, default=30, help="Top-k value for generating responses.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generating responses.")    

    args = parser.parse_args()
    model = Khaosz(args.model_dir).to(device='cuda', dtype=torch.bfloat16)
    
    processor(
        model,
        input_json_file=args.input_json_file,
        output_json_file=args.output_json_file,
        question_key=args.question_key,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )