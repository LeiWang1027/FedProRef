"""Backbone identity and feature-dimension helpers."""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path


_CHECKPOINT_SUFFIXES = {".bin", ".pt", ".pth", ".safetensors"}
_SHA256_RE = re.compile(r"(?<![0-9a-fA-F])([0-9a-fA-F]{64})(?![0-9a-fA-F])")
RN50_OPENAI_SHA256 = (
    "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762")


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _looks_like_checkpoint_path(value: str) -> bool:
    path = Path(os.path.expanduser(value))
    return (
        path.suffix.lower() in _CHECKPOINT_SUFFIXES
        or os.path.isabs(value)
        or value.startswith(".")
        or os.sep in value
    )


def resolve_pretrained_identity(backbone: str, pretrained: str) -> dict[str, str]:
    """Return a stable identity for local or registered OpenCLIP weights."""
    if not backbone or not pretrained:
        raise ValueError("backbone and pretrained must be nonempty")

    expanded = os.path.expanduser(pretrained)
    if os.path.isfile(expanded):
        real_path = os.path.realpath(expanded)
        return {
            "backbone": backbone,
            "pretrained": real_path,
            "source": real_path,
            "checkpoint_hash": _sha256_file(real_path),
        }
    if _looks_like_checkpoint_path(pretrained):
        raise FileNotFoundError(f"pretrained checkpoint not found: {pretrained}")

    import open_clip

    config = open_clip.get_pretrained_cfg(backbone, pretrained)
    if not config:
        raise ValueError(
            f"unknown OpenCLIP backbone/pretrained pair: {backbone}/{pretrained}")
    source = str(config.get("url") or config.get("hf_hub") or "")
    match = _SHA256_RE.search(source)
    if match is None:
        raise ValueError(
            f"OpenCLIP identity for {backbone}/{pretrained} has no stable SHA-256 source")
    return {
        "backbone": backbone,
        "pretrained": pretrained,
        "source": source,
        "checkpoint_hash": match.group(1).lower(),
    }


def open_clip_load_kwargs(identity: dict[str, str]) -> dict[str, bool]:
    """Return compatibility flags for a verified official TorchScript archive."""
    if (identity.get("backbone") == "RN50"
            and identity.get("checkpoint_hash") == RN50_OPENAI_SHA256):
        # PyTorch 2.6 defaults torch.load(weights_only=True), which rejects the
        # official OpenAI RN50 TorchScript archive. The exact official digest
        # above is verified before allowing OpenCLIP's legacy loader.
        return {"weights_only": False}
    return {}


def resolve_feature_dim(expected: int | None, train_features, test_features) -> int:
    """Resolve and validate the shared width of train and test feature matrices."""
    if getattr(train_features, "ndim", None) != 2 or getattr(test_features, "ndim", None) != 2:
        raise ValueError("train and test features must be rank-two matrices")
    if train_features.shape[0] == 0 or test_features.shape[0] == 0:
        raise ValueError("train and test feature matrices must be nonempty")

    train_dim = int(train_features.shape[1])
    test_dim = int(test_features.shape[1])
    if train_dim != test_dim:
        raise ValueError(
            f"train/test feature dimensions differ: train={train_dim}, test={test_dim}")
    if train_dim <= 0:
        raise ValueError("resolved feature dimension must be positive")
    if expected is not None:
        if int(expected) <= 0:
            raise ValueError("expected feature dimension must be positive")
        if int(expected) != train_dim:
            raise ValueError(
                f"expected feature dimension {int(expected)} but resolved {train_dim}")
    return train_dim
