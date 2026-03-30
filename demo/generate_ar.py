import os
import torch
from khaosz.config.param_config import ModelParameter
from khaosz.inference.core import disable_random_init
from khaosz.inference.generator import LoopGenerator, GenerationRequest


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def generate_text():

    with disable_random_init():
        model_dir = os.path.join(PROJECT_ROOT, "params")
        param = ModelParameter.load(model_dir)

    param.to(device="cuda", dtype=torch.bfloat16)
    query = input(">> ")

    request = GenerationRequest(
        query=query,
        temperature=0.8,
        top_p=0.95,
        top_k=50,
        max_len=param.config.max_len,
        history=None,
        system_prompt=None,
    )
    generator = LoopGenerator(param)
    response = generator.generate(request)

    print(response)


if __name__ == "__main__":
    generate_text()
