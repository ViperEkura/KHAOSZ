"""Parse the host-side runtime section of a serving configuration.

The Compose wrapper needs a few container-side values on the host as well:
``server.port`` (the port the container listens on) and ``server.device``
(used by the preflight GPU consistency check). Everything else under
``server:`` is owned by ``scripts/tools/server.py --config`` inside the
container.
"""

import argparse
import re
import shlex
from pathlib import Path
from typing import Dict

import yaml

ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _mapping(value, name: str) -> Dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _path(value, name: str, config_dir: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"runtime.paths.{name} is required")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return str(path.resolve())


def _port(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not 1 <= value <= 65535:
        raise ValueError(f"{name} must be between 1 and 65535")
    return value


def load_runtime(config_path: str) -> Dict[str, str]:
    path = Path(config_path).resolve()
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError("serving configuration must be a mapping")

    runtime = _mapping(config.get("runtime"), "runtime")
    if not runtime:
        raise ValueError("top-level runtime section is required")
    paths = _mapping(runtime.get("paths"), "paths")
    gpu = _mapping(runtime.get("gpu"), "gpu")
    container = _mapping(runtime.get("container"), "container")
    environment = _mapping(runtime.get("environment"), "environment")
    server = _mapping(config.get("server"), "server")

    job_name = runtime.get("job_name", "")
    if job_name and not isinstance(job_name, str):
        raise ValueError("runtime.job_name must be a string")
    if job_name and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", job_name):
        raise ValueError(
            "runtime.job_name must use letters, numbers, dot, underscore, or dash"
        )

    port = _port(runtime.get("port", 8000), "runtime.port")
    container_port = _port(server.get("port", 8000), "server.port")

    device = server.get("device", "cuda")
    if not isinstance(device, str) or not device.strip():
        raise ValueError("server.device must be a string")

    gpu_enabled = gpu.get("enabled", True)
    if not isinstance(gpu_enabled, bool):
        raise ValueError("runtime.gpu.enabled must be a boolean")

    devices = gpu.get("devices", "all")
    visible_devices = None
    if gpu_enabled:
        if devices == "all":
            pass
        elif isinstance(devices, list) and len(devices) == 1:
            text = str(devices[0])
            if not text.isdigit():
                raise ValueError(
                    "runtime.gpu.devices entries must be non-negative integers"
                )
            visible_devices = text
        else:
            raise ValueError(
                "runtime.gpu.devices must be 'all' or a single-device list such as [0]"
            )
    else:
        if devices != "all":
            raise ValueError(
                "runtime.gpu.devices is ignored when runtime.gpu.enabled is false"
            )
        if device != "cpu":
            raise ValueError(
                "server.device must be 'cpu' when runtime.gpu.enabled is false"
            )

    values = {
        "SERVE_JOB_NAME": job_name,
        "SERVE_PORT": str(port),
        "SERVE_CONTAINER_PORT": str(container_port),
        "SERVE_PARAM_DIR": _path(paths.get("param", "./params"), "param", path.parent),
        "SERVE_GPU_ENABLED": "true" if gpu_enabled else "false",
        "SERVE_DEVICE": device,
        "CUDA_TAG": str(container.get("cuda_tag", "cu130")),
    }
    if visible_devices is not None:
        values["CUDA_VISIBLE_DEVICES"] = visible_devices

    for name, value in environment.items():
        if not isinstance(name, str) or not ENV_NAME.fullmatch(name):
            raise ValueError(f"invalid runtime.environment name: {name!r}")
        if value is not None and not isinstance(value, (str, int, float, bool)):
            raise ValueError(f"runtime.environment.{name} must be a scalar")
    values["environment"] = environment
    return values


def shell_exports(runtime: Dict[str, str]) -> str:
    return "\n".join(
        f"export {name}={shlex.quote(value)}"
        for name, value in runtime.items()
        if name != "environment"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("exports", "environment"))
    parser.add_argument("config")
    args = parser.parse_args()

    try:
        runtime = load_runtime(args.config)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))

    if args.command == "exports":
        print(shell_exports(runtime))
        return
    for name, value in runtime["environment"].items():
        rendered = "" if value is None else str(value)
        print(f"{name}={rendered}", end="\0")


if __name__ == "__main__":
    main()
