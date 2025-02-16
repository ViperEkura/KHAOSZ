import torch
import json
import warnings
from module import Khaosz
warnings.filterwarnings("ignore")

def batch_generate(
    queries,
    temperature=0.95, 
    top_k=50, 
    top_p=0.8
    ):
    model = Khaosz("params")
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
        out_dict = {
            "query": query,
            "response": response
        }
        outputs.append(out_dict)
    return outputs

def dpo_generate(input_json_file):
    model = Khaosz("params")
    
    with open(input_json_file, "r") as f:
        json_file = json.loads(f)

    

    
if __name__  == "__main__":
     pass