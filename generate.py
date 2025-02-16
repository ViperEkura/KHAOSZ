import torch
import json
import warnings
from module import Khaosz
warnings.filterwarnings("ignore")

def batch_generate(
    queries,
    model_dir="params",
    temperature=0.95, 
    top_k=50, 
    top_p=0.8 
    ):
    model = Khaosz(model_dir)
    model = model.to(device='cuda', dtype=torch.float16)
    responses = [
        model.generate(
            query=query, 
            temperature=temperature, 
            top_k=top_k, 
            top_p=top_p
            ) for query in queries
        ]
    outputs = []
    for query, response in zip(queries, responses):
        outputs.append({"query": query, "response": response})
    return outputs

def dpo_generate(input_json_file):
    model = Khaosz("params")
    
    with open(input_json_file, "r") as f:
        json_file = json.loads(f)

    item_size = len(json_file)
    queries = [item["question"] for item in json_file]
    recepted = [item['recepted'] for item in json_file]
    del json_file
    
    rejected = batch_generate(queries)
    output_dict = []
    
    for i in range(item_size):
        output_dict.append({
            "question": queries[i],
            "recepted": recepted[i],
            "rejected": rejected[i]
        })

    return output_dict

def generate_dpo_data(dpo_dict):
    pass

    
if __name__  == "__main__":
     pass