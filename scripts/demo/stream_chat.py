from pathlib import Path

import torch
from astrai.inference import InferenceEngine
from astrai.model import AutoModel
from astrai.tokenize import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMETER_ROOT = Path(PROJECT_ROOT, "params")


def chat():
    model = AutoModel.from_pretrained(PARAMETER_ROOT)
    tokenizer = AutoTokenizer.from_pretrained(PARAMETER_ROOT)
    model.to(device="cuda", dtype=torch.bfloat16)

    messages = []
    engine = InferenceEngine(model=model, tokenizer=tokenizer)

    while True:
        query = input(">> ")
        if query == "!exit":
            break

        # Add user message
        messages.append({"role": "user", "content": query})

        # Generate response
        full_response = ""
        prompt = tokenizer.apply_chat_template(messages, tokenize=False)

        for token in engine.generate(
            prompt=prompt,
            stream=True,
            max_tokens=model.config.max_len,
            temperature=0.8,
            top_p=0.95,
            top_k=50,
        ):
            print(token, end="", flush=True)
            full_response += token

        print()
        # Add assistant response to messages
        messages.append({"role": "assistant", "content": full_response.strip()})


if __name__ == "__main__":
    chat()
