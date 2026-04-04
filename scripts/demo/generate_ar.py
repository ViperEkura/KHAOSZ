from pathlib import Path

import torch

from astrai.config.param_config import ModelParameter
from astrai.inference import InferenceEngine

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMETER_ROOT = Path(PROJECT_ROOT, "params")


def generate_text():
    param = ModelParameter.load(PARAMETER_ROOT, disable_init=True)
    param.to(device="cuda", dtype=torch.bfloat16)

    query = input(">> ")

    engine = InferenceEngine(param)
    response = engine.generate(
        prompt=query,
        stream=False,
        max_tokens=param.config.max_len,
        temperature=0.8,
        top_p=0.95,
        top_k=50,
    )

    print(response)


if __name__ == "__main__":
    generate_text()
