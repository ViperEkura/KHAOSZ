import os
from huggingface_hub import snapshot_download


if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    snapshot_download(
        repo_id="ViperEk/KHAOSZ",
        local_dir=script_dir,  
        force_download=True
    )