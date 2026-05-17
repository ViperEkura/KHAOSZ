from pathlib import Path

import torch

from astrai.inference import InferenceEngine
from astrai.model import AutoModel
from astrai.tokenize import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMETER_ROOT = Path(PROJECT_ROOT, "params")


def generate_text():
    model = AutoModel.from_pretrained(PARAMETER_ROOT)
    tokenizer = AutoTokenizer.from_pretrained(PARAMETER_ROOT)
    model.to(device="cuda", dtype=torch.bfloat16)

    query = input(">> ")

    engine = InferenceEngine(
        model=model,
        tokenizer=tokenizer,
    )
    for token in engine.generate(
        prompt=query,
        stream=True,
        max_tokens=2048,
        temperature=0.8,
        top_p=0.95,
        top_k=50,
    ):
        print(token, end="", flush=True)


if __name__ == "__main__":
    generate_text()
