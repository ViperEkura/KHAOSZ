import torch
from pathlib import Path
from astrai.config.param_config import ModelParameter
from astrai.inference.core import disable_random_init
from astrai.inference.generator import GeneratorFactory, GenerationRequest

PROJECT_ROOT = Path(__file__).parent.parent
PARAMETER_ROOT = Path(PROJECT_ROOT, "params")


def batch_generate():

    with disable_random_init():
        param = ModelParameter.load(PARAMETER_ROOT)
        param.to(device="cuda", dtype=torch.bfloat16)

    inputs = [
        "你好",
        "请问什么是人工智能",
        "今天天气如何",
        "我感到焦虑， 请问我应该怎么办",
        "请问什么是显卡",
    ]

    request = GenerationRequest(
        query=inputs,
        temperature=0.8,
        top_p=0.95,
        top_k=50,
        max_len=param.config.max_len,
        history=None,
        system_prompt=None,
    )
    generator = GeneratorFactory.create(param, request)
    responses = generator.generate(request)

    for q, r in zip(inputs, responses):
        print((q, r))


if __name__ == "__main__":
    batch_generate()
