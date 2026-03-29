import os
import torch
from khaosz.config.param_config import ModelParameter
from khaosz.inference.core import disable_random_init
from khaosz.inference.generator import StreamGenerator, GenerationRequest


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))

def chat():
    
    with disable_random_init():
        model_dir = os.path.join(PROJECT_ROOT, "params")
        param = ModelParameter.load(model_dir)

    param.to(device='cuda', dtype=torch.bfloat16)
    generator = StreamGenerator(param)

    history = []
    while True:
        query = input(">> ")
        if query == "!exit":
            break
        
        request = GenerationRequest(
            query=query,
            temperature=0.8,
            top_p=0.95,
            top_k=50,
            max_len=param.config.max_len,
            history=history,
            system_prompt=None,
        )
        
        response_size = 0
        full_response = ""
        for response in generator.generate(request):
            # response is the cumulative response up to current token
            print(response[response_size:], end="", flush=True)
            response_size = len(response)
            full_response = response
        
        # After generation, update history
        history.append((query, full_response.strip()))


if __name__ == "__main__":
    chat()