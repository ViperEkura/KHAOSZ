import os
import torch
from khaosz import Khaosz


def batch_generate():
    script_dir = os.path.dirname(__file__)
    model_dir = os.path.join(script_dir, "params")
    model = Khaosz(model_dir).to(device='cuda', dtype=torch.bfloat16)
    
    inputs = ["你好", "请问什么是人工智能", "今天天气如何", "我感到焦虑， 请问我应该怎么办", "请问什么是显卡"]

    responses = model.batch_generate(
        queries=inputs,
        temperature=0.95,
        top_p=0.95,
        top_k=50
    )
    
    for q, r in zip(inputs, responses):
        print((q, r))

if __name__ == "__main__":
    batch_generate()