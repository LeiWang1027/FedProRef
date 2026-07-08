"""
FedProRef: Utility functions
- Evaluation metrics: ECE, NLL, Accuracy
- Seeding, logging helpers
"""
import os
import sys
import random
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Metrics ──────────────────────────────────────────────────────────

def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Top-1 accuracy from raw logits."""
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def compute_ece(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """Expected Calibration Error (ECE)."""
    probs = F.softmax(logits, dim=1)
    confidences, preds = probs.max(dim=1)
    accuracies = preds.eq(labels)

    ece = 0.0
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            avg_conf = confidences[mask].mean().item()
            avg_acc = accuracies[mask].float().mean().item()
            ece += mask.sum().item() * abs(avg_acc - avg_conf)
    ece /= len(labels)
    return ece


def compute_nll(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Negative log-likelihood."""
    return F.cross_entropy(logits, labels).item()


def evaluate_all(model, features: torch.Tensor, labels: torch.Tensor, device: str):
    """Return dict with acc, ece, nll."""
    model.eval()
    with torch.no_grad():
        features, labels = features.to(device), labels.to(device)
        logits = model(features)
    return {
        "acc": compute_accuracy(logits, labels) * 100,
        "ece": compute_ece(logits, labels),
        "nll": compute_nll(logits, labels),
    }


# ── Logger ───────────────────────────────────────────────────────────

class _TeeStream:
    """Write to both the original stream and a file simultaneously."""
    def __init__(self, original, file_obj):
        self._orig = original
        self._file = file_obj

    def write(self, data):
        self._orig.write(data)
        self._orig.flush()
        self._file.write(data)
        self._file.flush()

    def flush(self):
        self._orig.flush()
        self._file.flush()

    def isatty(self):
        return self._orig.isatty()


class Logger:
    def __init__(self, log_dir: str, exp_name: str):
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(log_dir, f"{exp_name}_{ts}.log")
        self.records = []

        # Open log file and tee stdout/stderr to it
        self._log_file = open(self.path, "a", encoding="utf-8", buffering=1)
        sys.stdout = _TeeStream(sys.__stdout__, self._log_file)
        sys.stderr = _TeeStream(sys.__stderr__, self._log_file)

    def log_params(self, args):
        """Write all experiment parameters to the top of the log file."""
        sep = "=" * 70
        lines = [sep, f"  Experiment Log — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", sep]
        for k, v in sorted(vars(args).items()):
            lines.append(f"  {k:<30s} = {v}")
        lines.append(sep)
        block = "\n".join(lines) + "\n"
        print(block)   # goes through TeeStream → terminal + file

    def log(self, rnd: int, metrics: dict):
        line = f"Round {rnd:>4d} | " + " | ".join(
            f"{k}: {v:.4f}" for k, v in metrics.items()
        )
        print(line)    # TeeStream handles writing to file
        self.records.append({"round": rnd, **metrics})

    def best(self, key="acc"):
        vals = [r[key] for r in self.records]
        if key == "acc":
            idx = int(np.argmax(vals))
        else:
            idx = int(np.argmin(vals))
        return self.records[idx]

    def close(self):
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        self._log_file.close()
