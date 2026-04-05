import argparse
from pathlib import Path

import torch

from astrai.inference.server import run_server


def main():
    parser = argparse.ArgumentParser(description="Start AstrAI inference HTTP server")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host address (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port number (default: 8000)"
    )
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--param-path",
        type=Path,
        default=None,
        help="Path to model parameters (default: project_root/params)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to load model on (default: cuda)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Data type for model weights (default: bfloat16)",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=16,
        help="Maximum batch size for continuous batching (default: 16)",
    )
    args = parser.parse_args()

    # Convert dtype string to torch dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    project_root = Path(__file__).parent.parent.parent
    param_path = args.param_path or (project_root / "params")
    print(f"Starting AstrAI inference server on http://{args.host}:{args.port}")
    print(f"Model parameters expected at: {param_path}")
    print(f"Device: {args.device}, Dtype: {args.dtype}")
    run_server(
        host=args.host,
        port=args.port,
        reload=args.reload,
        device=args.device,
        dtype=dtype,
        param_path=param_path,
        max_batch_size=args.max_batch_size,
    )


if __name__ == "__main__":
    main()
