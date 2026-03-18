import torch
import json
import torch
import argparse

from khaosz import Khaosz
from typing import List
from tqdm import tqdm


def batch_generate(
    model: Khaosz,
    query: List[str],
    temperature: float, 
    top_k: int, 
    top_p: float,
    batch_size: int,
) -> List:
    assert batch_size > 0
    sorted_query = sorted(query, key=lambda x: len(x), reverse=True)
    original_indices = {query: idx for idx, query in enumerate(query)}
    
    responses = [None] * len(query) 
    total_batches = (len(sorted_query) + batch_size - 1) // batch_size 
    
    for i in tqdm(range(0, total_batches * batch_size, batch_size), desc="Generating responses"):
        batch_query = sorted_query[i: min(i + batch_size, len(query))]
        if not isinstance(batch_query, list):
            batch_query = [batch_query]
        
        batch_responses = model.batch_generate(
            query=batch_query,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        
        for query, response in zip(batch_query, batch_responses):
            original_idx = original_indices[query] 
            responses[original_idx] = response  

    return responses


def processor(
    model_dir: str,
    input_json_file: str,
    output_json_file: str,
    batch_size: int,
    temperature: float,
    top_k: int,
    top_p: float,
    question_key: str,
    response_key: str,
):
    model = Khaosz(model_dir).to(device='cuda', dtype=torch.bfloat16)
    
    with open(input_json_file, "r", encoding='utf-8') as f:
        input_data = [json.loads(line) for line in f]
        query = [item[question_key] for item in input_data]
    
    responses = batch_generate(
        model=model,
        query=query,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        batch_size=batch_size
    )
    
    # Write output in JSONL format
    with open(output_json_file, "w", encoding='utf-8') as f:
        for query, response in zip(query, responses):
            output_item = {question_key: query, response_key: response}
            f.write(json.dumps(output_item, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run generate with a Khaosz model.")
    
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the model directory.")
    parser.add_argument("--input_json_file", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output_json_file", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--question_key", type=str, default="question", help="Key for the question in the input JSON.")
    parser.add_argument("--response_key", type=str, default="response", help="Key for the response in the output JSON.")
    parser.add_argument("--temperature", type=float, default=0.60, help="Temperature for generating responses.")
    parser.add_argument("--top_k", type=int, default=30, help="Top-k value for generating responses.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p value for generating responses.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generating responses.")    

    args = parser.parse_args()
    processor(**vars(args))