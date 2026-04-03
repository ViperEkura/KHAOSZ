from pathlib import Path

import torch

from astrai.config.param_config import ModelParameter
from astrai.inference.generator import GenerationRequest, GeneratorFactory

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMETER_ROOT = Path(PROJECT_ROOT, "params")


def chat():
    param = ModelParameter.load(PARAMETER_ROOT, disable_init=True)
    param.to(device="cuda", dtype=torch.bfloat16)

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
            stream=True,
        )
        generator = GeneratorFactory.create(param, request)

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
