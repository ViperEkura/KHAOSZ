import os
from huggingface_hub import snapshot_download


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    snapshot_download(
        repo_id="ViperEk/KHAOSZ",
        local_dir=os.path.join(PROJECT_ROOT, "params"),  
        force_download=True
    )