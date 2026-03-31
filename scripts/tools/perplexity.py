import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import tqdm

from torch import Tensor
from astrai.config.param_config import ModelParameter


def compute_perplexity(
    model: nn.Module,
    input_ids: Tensor,
    input_mask: Tensor,
) -> Tensor:
    """
    Compute the perplexity of a batch of input sequences,
    where PPL = exp(-(1/N) * sum(log P(w_i | w_<i))).
    """

    output = model(input_ids, input_mask)
    logits = output["logits"]

    shifted_logits = logits[:, :-1, :]  # [batch_size, seq_len-1, vocab_size]
    shifted_input_ids = input_ids[:, 1:]  # [batch_size, seq_len-1]
    shifted_mask = input_mask[:, 1:]  # [batch_size, seq_len-1]

    loss = F.cross_entropy(
        shifted_logits.flatten(0, 1), shifted_input_ids.flatten(0, 1), reduction="none"
    )

    loss = loss.view(shifted_input_ids.shape)  # [batch_size, seq_len-1]
    loss = loss * shifted_mask
    sentence_loss = (loss).sum(dim=1) / shifted_mask.sum(dim=1)
    perplexity = torch.exp(sentence_loss)  # [batch_size]

    return perplexity


def process_file(
    model_dir: str, input_file: str, output_file: str, batch_size: int, text_key: str
):
    param = ModelParameter.load(model_dir, disable_init=True)
    param.to(device="cuda", dtype=torch.bfloat16)
    model = param.model
    tokenizer = param.tokenizer

    with open(input_file, "r", encoding="utf-8") as f:
        input_data = [json.loads(line) for line in f]

    texts = [item[text_key] for item in input_data]
    encoded_texts = [tokenizer.encode(text) for text in texts]
    output_data = []

    for i in tqdm(
        range(0, len(encoded_texts), batch_size), desc="Computing perplexity"
    ):
        batch_encoded = encoded_texts[i : i + batch_size]
        batch_texts = texts[i : i + batch_size]
        max_len = max(len(seq) for seq in batch_encoded)
        padded_ids = []
        masks = []

        for seq in batch_encoded:
            pad_len = max_len - len(seq)
            padded_seq = [tokenizer.pad_id] * pad_len + seq
            mask = [False] * pad_len + [True] * len(seq)
            padded_ids.append(padded_seq)
            masks.append(mask)

        input_ids = torch.tensor(padded_ids, device="cuda", dtype=torch.long)
        input_mask = torch.tensor(masks, device="cuda", dtype=torch.bool)
        perplexity = compute_perplexity(model, input_ids, input_mask)

        for text, ppl in zip(batch_texts, perplexity):
            output_data.append({text_key: text, "ppl": float(ppl.item())})

    with open(output_file, "w", encoding="utf-8") as f:
        for item in output_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run perplexity with a Khaosz model.")
    parser.add_argument(
        "--model_dir", type=str, required=True, help="Path to the model directory."
    )
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to the input file."
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Path to the output file."
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for evaluation."
    )
    parser.add_argument(
        "--text_key",
        type=str,
        default="text",
        help="Key for the text field in the input data.",
    )
    args = parser.parse_args()

    with torch.inference_mode():
        process_file(**vars(args))
