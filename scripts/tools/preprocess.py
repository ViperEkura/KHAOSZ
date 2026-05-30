"""CLI: raw JSONL → tokenized .h5/.bin via Pipeline."""

import argparse
import sys

from astrai.preprocess import Pipeline, detect_format


def main():
    parser = argparse.ArgumentParser(
        description="Raw JSONL → tokenized .h5/.bin for training"
    )
    parser.add_argument(
        "inputs", nargs="+", metavar="JSONL", help="One or more JSONL files"
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        required=True,
        help="Output directory (domain subdirs auto-created)",
    )
    parser.add_argument(
        "--tokenizer_path",
        default="params",
        help="Path to tokenizer (default: params)",
    )
    parser.add_argument(
        "--text_key",
        default=None,
        help="JSON key for text (auto-detect if omitted)",
    )
    parser.add_argument(
        "--domain_key",
        default=None,
        help="JSON key for domain label (auto-detect if omitted)",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=2048,
        help="Max token length per doc (default: 2048)",
    )
    parser.add_argument(
        "--min_text_len",
        type=int,
        default=50,
        help="Min chars per doc (default: 50)",
    )
    parser.add_argument(
        "--max_text_len",
        type=int,
        default=2_000_000,
        help="Max chars per doc (default: 2000000)",
    )
    parser.add_argument(
        "--no_dedup",
        action="store_true",
        help="Skip exact dedup",
    )
    parser.add_argument(
        "--max_items",
        type=int,
        default=None,
        help="Max docs to process (default: all)",
    )
    parser.add_argument(
        "--max_tokens_per_shard",
        type=int,
        default=100_000_000,
        help="Max tokens per .h5 shard (default: 100M)",
    )
    parser.add_argument(
        "--format",
        dest="storage_format",
        choices=["h5", "bin"],
        default="bin",
        help="Output format (default: bin)",
    )
    parser.add_argument(
        "--detect",
        action="store_true",
        help="Detect and print JSONL schema, then exit",
    )
    args = parser.parse_args()

    if args.detect:
        fmt = detect_format(args.inputs)
        print(f"text key   : {fmt['text_key']}")
        print(f"domain key : {fmt['domain_key']}")
        print(f"chat mode  : {fmt['is_chat']}")
        sys.exit(0)

    Pipeline(
        input_paths=args.inputs,
        output_dir=args.output_dir,
        tokenizer_path=args.tokenizer_path,
        text_key=args.text_key,
        domain_key=args.domain_key,
        max_len=args.max_len,
        min_text_len=args.min_text_len,
        max_text_len=args.max_text_len,
        dedup=not args.no_dedup,
        max_items=args.max_items,
        max_tokens_per_shard=args.max_tokens_per_shard,
        storage_format=args.storage_format,
    ).run()


if __name__ == "__main__":
    main()
