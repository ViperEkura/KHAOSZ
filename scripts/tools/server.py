import argparse
from pathlib import Path

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
    args = parser.parse_args()

    # If param_path is provided, set environment variable or modify global?
    # Currently the server loads from default location on startup.
    # We could pass it via an environment variable, but for simplicity we assume
    # the default location is correct.
    project_root = Path(__file__).parent.parent
    param_path = args.param_path or (project_root / "params")
    print(f"Starting AstrAI inference server on http://{args.host}:{args.port}")
    print(f"Model parameters expected at: {[param_path]}")
    run_server(host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
