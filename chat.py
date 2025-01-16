import torch
from module import Khaosz, Transformer, Config

def chat():
    model = Khaosz("params")
    model = model.to(device='cuda', dtype=torch.bfloat16)
    response_size = 0
    histroy = []

    while True:
        query = input(">> ")
        if query == "!exit":
            break
        
        for response, histroy in model.stream_generate(
            query=query, 
            history=histroy,
            temperature=0.9,    
        ):
            print(response[response_size:], end="")
            response_size = len(response)
            
        print("")

def test():
    cfg = Config("params/config.json")
    model = Transformer(cfg)
    print(model)
    print(f"parameter size: {model.parameter_size():,}")

if __name__ == "__main__":
    test()