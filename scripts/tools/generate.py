import argparse
import json

import torch

from astrai.model import AutoModel
from astrai.tokenize import AutoTokenizer
from astrai.inference import InferenceEngine


def processor(
    model_dir: str,
    input_json_file: str,
    output_json_file: str,
    temperature: float,
    top_k: int,
    top_p: float,
    question_key: str,
    response_key: str,
):
    # Load model using AutoModel
    model = AutoModel.from_pretrained(model_dir, device="cuda", dtype=torch.bfloat16)
    engine = InferenceEngine(model=model.model, tokenizer=model.tokenizer)

    with open(input_json_file, "r", encoding="utf-8") as f:
        input_data = [json.loads(line) for line in f]

    queries = [item[question_key] for item in input_data]

    responses = engine.generate(
        prompt=queries,
        stream=False,
        max_tokens=model.config.max_len,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )

    with open(output_json_file, "w", encoding="utf-8") as f:
        for query, response in zip(queries, responses):
            output_item = {question_key: query, response_key: response}
            f.write(json.dumps(output_item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run generate with a Khaosz model.")

    parser.add_argument(
        "--model_dir", type=str, required=True, help="Path to the model directory."
    )
    parser.add_argument(
        "--input_json_file",
        type=str,
        required=True,
        help="Path to the input JSONL file.",
    )
    parser.add_argument(
        "--output_json_file",
        type=str,
        required=True,
        help="Path to the output JSONL file.",
    )
    parser.add_argument(
        "--question_key",
        type=str,
        default="question",
        help="Key for the question in the input JSON.",
    )
    parser.add_argument(
        "--response_key",
        type=str,
        default="response",
        help="Key for the response in the output JSON.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.60,
        help="Temperature for generating responses.",
    )
    parser.add_argument(
        "--top_k", type=int, default=30, help="Top-k value for generating responses."
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p value for generating responses.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for generating responses."
    )

    args = parser.parse_args()

    with torch.inference_mode():
        processor(**vars(args))
