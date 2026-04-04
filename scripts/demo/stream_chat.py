from pathlib import Path

import torch

from astrai.config.param_config import ModelParameter
from astrai.inference import InferenceEngine

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMETER_ROOT = Path(PROJECT_ROOT, "params")


def chat():
    param = ModelParameter.load(PARAMETER_ROOT, disable_init=True)
    param.to(device="cuda", dtype=torch.bfloat16)

    history = []
    engine = InferenceEngine(param)

    while True:
        query = input(">> ")
        if query == "!exit":
            break

        full_response = ""

        for token in engine.generate(
            prompt=query,
            stream=True,
            max_tokens=param.config.max_len,
            temperature=0.8,
            top_p=0.95,
            top_k=50,
        ):
            print(token, end="", flush=True)
            full_response += token

        print()
        history.append((query, full_response.strip()))


if __name__ == "__main__":
    chat()
