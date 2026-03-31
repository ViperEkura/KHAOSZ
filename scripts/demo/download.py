from pathlib import Path
from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).parent.parent
PARAMETER_ROOT = Path(PROJECT_ROOT, "params")

if __name__ == "__main__":
    snapshot_download(
        repo_id="ViperEk/KHAOSZ",
        local_dir=PARAMETER_ROOT,
        force_download=True,
    )
