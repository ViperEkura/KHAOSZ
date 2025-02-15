import torch
import json
import warnings
from module import Khaosz
warnings.filterwarnings("ignore")

def batch_generate():
    model = Khaosz("params")
    model = model.to(device='cuda', dtype=torch.float16)
    queries = ["你会做什么", "什么是人工智能", "NLP技术包括NLG等等方面么" ]
    responses = model.batch_generate(
        queries=queries,
        temperature=0.7,
        top_k=50,
        top_p=0.8,
    )
    out_info = []
    for query, response in zip(queries, responses):
        out_dict = {
            "query": query,
            "response": response
        }
        out_info.append(out_dict)
        print(out_dict)


def generate_dpo_data(data_path, question_key, recepted_key, rejected_key, batch_size):
    model = Khaosz("params")
    model = model.to(device='cuda', dtype=torch.float16)
    
    with open(data_path, "r") as f:
        json_file = json.loads(f)    
        questions = [json_file[i][question_key] for i in range(len(json_file))]
        recepted = [json_file[i][recepted_key] for i in range(len(json_file))]
    
    rejected = []
    pass
    
    
if __name__  == "__main__":
     batch_generate()