import os
import json
import torch
import warnings

from module import Khaosz
from typing import List, Dict

warnings.filterwarnings("ignore")

def batch_generate(
    queries,
    model: Khaosz,
    temperature=0.95, 
    top_k=50, 
    top_p=0.8 
) -> Dict:
    responses = [
        model.generate(
            query=query, 
            temperature=temperature, 
            top_k=top_k, 
            top_p=top_p
            ) for query in queries
        ]
    outputs = []
    for response in responses:
        outputs.append(response)
    return outputs

def dpo_generate(
    model: Khaosz,
    input_json_file: str,
    question_key: str="question",
    recepted_key: str="recepted",
)-> List:  
    with open(input_json_file, "r") as f:
        json_file = json.loads(f)
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
        json.dump(output_dict, f)


if __name__  == "__main__":
     model = Khaosz("params")
     model = model.to(device='cuda', dtype=torch.float16)
     
     dpo(
         model,
         input_json_file="data/dpo_data.json",
         output_json_file="data/dpo_output.json",
         question_key="question",
         recepted_key="recepted"
     )