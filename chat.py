import torch
import warnings
from module import Khaosz, Transformer, Config

warnings.filterwarnings("ignore")

def chat():
    model = Khaosz("params")
    model = model.to(device='cuda', dtype=torch.bfloat16)
    histroy = []
    with torch.no_grad():
        while True:
            query = input(">> ")
            if query == "!exit":
                break
            
            response_size = 0
            for response, histroy in model.stream_generate(
                query=query, 
                history=histroy,
                temperature=1.2,
                top_p=0.3
            ):
                print(response[response_size:], end="", flush=True)
                response_size = len(response)       
            print()


def show_parameter_size():
    cfg = Config("params/config.json")
    model = Transformer(cfg)
    print(f"parameter size: {model.parameter_size():,}")
    

if __name__ == "__main__":
    chat()