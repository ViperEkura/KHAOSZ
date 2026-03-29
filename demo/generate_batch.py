import os
import torch
from khaosz.config.param_config import ModelParameter
from khaosz.inference.core import disable_random_init
from khaosz.inference.generator import BatchGenerator, GenerationRequest

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))

def batch_generate():
    with disable_random_init():
        model_dir = os.path.join(PROJECT_ROOT, "params")
        param = ModelParameter.load(model_dir)

    param.to(device='cuda', dtype=torch.bfloat16)
    generator = BatchGenerator(param)
    inputs = ["你好", "请问什么是人工智能", "今天天气如何", "我感到焦虑， 请问我应该怎么办", "请问什么是显卡"]
    
    request = GenerationRequest(
        query=inputs,
        temperature=0.8,
        top_p=0.95,
        top_k=50,
        max_len=param.config.max_len,
        history=None,
        system_prompt=None,
    )
    responses = generator.generate(request)
    
    for q, r in zip(inputs, responses):
        print((q, r))

if __name__ == "__main__":
    batch_generate()