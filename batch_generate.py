import torch
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
    
if __name__  == "__main__":
     batch_generate()