import os
import torch
from khaosz import Khaosz


def generate_text():
    script_dir = os.path.dirname(__file__)
    model_dir = os.path.join(script_dir, "params")
    model = Khaosz(model_dir).to(device='cuda', dtype=torch.bfloat16)

    query = input(">> ")
    
    response = model.text_generate(
        query=query, 
        temperature=0.6,
        top_p=0.95,
        top_k=30
    )
    
    print(response)

    

if __name__ == "__main__":
    generate_text()