import os
import torch
from khaosz import Khaosz


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))

def chat():
    model_dir = os.path.join(PROJECT_ROOT, "params")
    model = Khaosz(model_dir).to(device='cuda', dtype=torch.bfloat16)

    histroy = []
    while True:
        query = input(">> ")
        if query == "!exit":
            break
        
        response_size = 0
        for response, histroy in model.stream_generate(
            query=query, 
            history=histroy,
            temperature=0.8,
            top_p=0.95,
            top_k=50
        ):
            print(response[response_size:], end="", flush=True)
            response_size = len(response)
    

if __name__ == "__main__":
    chat()