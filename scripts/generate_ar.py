import os
import torch
from khaosz import Khaosz


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))

def generate_text():
    model_dir = os.path.join(PROJECT_ROOT, "params")
    model = Khaosz(model_dir).to(device='cuda', dtype=torch.bfloat16)

    query = input(">> ")
    
    response = model.text_generate(
        query=query, 
        temperature=0.8,
        top_p=0.95,
        top_k=50
    )
    
    print(response)

    

if __name__ == "__main__":
    generate_text()