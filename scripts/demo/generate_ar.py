import torch
from pathlib import Path
from astrai.config.param_config import ModelParameter
from astrai.inference.generator import GeneratorFactory, GenerationRequest

PROJECT_ROOT = Path(__file__).parent.parent
PARAMETER_ROOT = Path(PROJECT_ROOT, "params")


def generate_text():
    param = ModelParameter.load(PARAMETER_ROOT, disable_init=True)
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
    generator = GeneratorFactory.create(param, request)
    response = generator.generate(request)

    print(response)


if __name__ == "__main__":
    generate_text()
