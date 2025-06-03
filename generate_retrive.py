from module import Khaosz
import torch


if __name__ == "__main__":
    model = Khaosz("params")
    model = model.to(dtype=torch.bfloat16, device="cuda")
    query = "什么是b站"
    
    resp = model.retrieve_generate(query, top_k=50, top_p=0.98)
    print(resp)
     
     