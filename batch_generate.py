import torch
import json
import warnings
from module import Khaosz
warnings.filterwarnings("ignore")

def batch_generate():
    model = Khaosz("params")
    model = model.to(device='cuda', dtype=torch.float16)
    queries = ["什么是人工智能", "什么是高性能计算", "什么是transformer"]
    responses = model.batch_generate(
        queries=queries,
        temperature=0.9,
        top_k=50,
        top_p=0.6,
    )
    for response in responses:
        print(response)

def generate_dpo_data(data_path, question_key, recepted_key, rejected_key, batch_size):
    model = Khaosz("params")
    model = model.to(device='cuda', dtype=torch.float16)
    
    with open(data_path, "r") as f:
        json_file = json.loads(f)
    questions = [json_file[i][question_key] for i in range(len(json_file))]
    
    pass
    
    
if __name__  == "__main__":
     batch_generate()