#!/usr/bin/env python3
# prefetch_ckpt.py
import os
import argparse
from huggingface_hub import snapshot_download

def main():
    cache_dir = os.getenv("HF_HOME", "/data/data_team/cache/huggingface")
    snapshot_download(
        repo_id="Qwen/Qwen3-235B-A22B-Thinking-2507",
        cache_dir=cache_dir,
        resume_download=True,
    )

if __name__ == "__main__":
    main()