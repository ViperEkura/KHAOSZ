import torch
import warnings
from module import Khaosz, Transformer, Config

warnings.filterwarnings(
    "ignore",
    message=".*Torch was not compiled with flash attention.*",
    category=UserWarning,
    module='torch.*'
)

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
            temperature=1.0,
            top_k=10,
            top_p=0.8
        ):
            print(response[response_size:], end="")
            response_size = len(response)
            
        print("")

def show_parameter_size():
    cfg = Config("params/config.json")
    model = Transformer(cfg)
    print(f"parameter size: {model.parameter_size():,}")
    

if __name__ == "__main__":
    chat()