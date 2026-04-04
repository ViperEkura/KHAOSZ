from pathlib import Path

import torch

from astrai.config.param_config import ModelParameter
from astrai.inference import InferenceEngine

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMETER_ROOT = Path(PROJECT_ROOT, "params")


def batch_generate():
    param = ModelParameter.load(PARAMETER_ROOT, disable_init=True)
    param.to(device="cuda", dtype=torch.bfloat16)

    inputs = [
        "你好",
        "请问什么是人工智能",
        "今天天气如何",
        "我感到焦虑， 请问我应该怎么办",
        "请问什么是显卡",
    ]

    engine = InferenceEngine(param)
    responses = engine.generate(
        prompt=inputs,
        stream=False,
        max_tokens=param.config.max_len,
        temperature=0.8,
        top_p=0.95,
        top_k=50,
    )

    for q, r in zip(inputs, responses):
        print((q, r))


if __name__ == "__main__":
    batch_generate()
