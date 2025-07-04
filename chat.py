import os
import torch
from khaosz import Khaosz


def chat():
    model_dir = os.path.join(os.path.dirname(__file__), "params")
    model = Khaosz(model_dir)
    model = model.to(device='cuda', dtype=torch.bfloat16)
    histroy = []
    while True:
        query = input(">> ")
        if query == "!exit":
            break
        
        response_size = 0
        for response, histroy in model.stream_generate(
            query=query, 
            history=histroy,
            temperature=0.6,
            top_p=0.95,
            top_k=30
        ):
            print(response[response_size:], end="", flush=True)
            response_size = len(response)
    

if __name__ == "__main__":
    chat()