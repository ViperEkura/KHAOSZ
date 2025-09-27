import os
import torch
from khaosz import Khaosz


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))

def batch_generate():
    model_dir = os.path.join(PROJECT_ROOT, "params")
    model = Khaosz(model_dir).to(device='cuda', dtype=torch.bfloat16)
    inputs = ["你好", "请问什么是人工智能", "今天天气如何", "我感到焦虑， 请问我应该怎么办", "请问什么是显卡"]
    
    responses = model.batch_generate(
        queries=inputs,
        temperature=0.7,
        top_p=0.95,
        top_k=30
    )
    
    for q, r in zip(inputs, responses):
        print((q, r))

if __name__ == "__main__":
    batch_generate()