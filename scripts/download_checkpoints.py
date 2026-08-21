#!/usr/bin/env python3
"""Download and verify the exact public checkpoints used by FedProRef."""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path


CHECKPOINTS = {
    "vit-b-16": {
        "filename": "old_open_clip_model.safetensors",
        "url": (
            "https://huggingface.co/timm/"
            "vit_base_patch16_clip_224.openai/resolve/main/"
            "open_clip_model.safetensors"
        ),
        "sha256": "4b8699299b1e8997753c64b052ba32031449d5d853f55a039148560ee02b820f",
    },
    "rn50": {
        "filename": "RN50_openai.pt",
        "url": (
            "https://openaipublic.azureedge.net/clip/models/"
            "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/"
            "RN50.pt"
        ),
        "sha256": "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_checkpoint(name: str, output_dir: Path) -> Path:
    spec = CHECKPOINTS[name]
    target = output_dir / spec["filename"]
    if target.exists():
        observed = sha256_file(target)
        if observed != spec["sha256"]:
            raise RuntimeError(
                f"Refusing to overwrite {target}: SHA-256 is {observed}, "
                f"expected {spec['sha256']}"
            )
        print(f"[verified] {target}")
        return target

    output_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{spec['filename']}.", suffix=".part", dir=output_dir
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        print(f"[download] {spec['url']}")
        urllib.request.urlretrieve(spec["url"], temporary)
        observed = sha256_file(temporary)
        if observed != spec["sha256"]:
            raise RuntimeError(
                f"Downloaded {name} SHA-256 is {observed}, expected {spec['sha256']}"
            )
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"[saved] {target}")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        choices=["all", *CHECKPOINTS],
        default="all",
        help="Checkpoint to download (default: all).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "pretrain_path",
        help="Destination directory (default: repository/pretrain_path).",
    )
    args = parser.parse_args()
    names = CHECKPOINTS if args.checkpoint == "all" else [args.checkpoint]
    for name in names:
        download_checkpoint(name, args.output_dir.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
