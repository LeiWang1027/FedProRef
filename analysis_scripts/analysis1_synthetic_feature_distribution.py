#!/usr/bin/env python3
"""Offline synthetic-feature distribution analysis for FedProRef.

This script is intentionally strict: it computes metrics only when the exact
feature cache, final MLP refiner state, and merged prototype pool are provided.
If any required artifact is missing or incompatible, it writes artifact_audit.json
and exits without producing placeholder metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from refiner_mlp import MLPRefiner  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FedProRef synthetic feature distribution analysis")
    p.add_argument("--dataset", default="cifar100")
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--partition_seed", type=int, default=42)
    p.add_argument("--feature-cache", required=True)
    p.add_argument("--artifact", default=None,
                   help="Combined FedProRef analysis artifact saved by federated_loop.py")
    p.add_argument("--refiner-checkpoint", default=None,
                   help="Standalone final MLP refiner state_dict, if not using --artifact")
    p.add_argument("--prototype-pool", default=None,
                   help="Standalone merged prototype pool, if not using --artifact")
    p.add_argument("--feat-dim", type=int, default=512)
    p.add_argument("--num-classes", type=int, default=100)
    p.add_argument("--refiner-hidden", type=int, default=512)
    p.add_argument("--refiner-layers", type=int, default=3)
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--samples-per-class", type=int, default=100)
    p.add_argument("--analysis-seed", type=int, default=20260709)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", default="results/analysis1_synthetic_distribution")
    return p.parse_args()


def file_record(path: str | None) -> dict[str, Any]:
    if not path:
        return {"path": None, "exists": False}
    p = Path(path)
    if not p.exists():
        return {"path": str(p.resolve()), "exists": False}
    st = p.stat()
    return {
        "path": str(p.resolve()),
        "exists": True,
        "size_bytes": st.st_size,
        "mtime": st.st_mtime,
    }


def json_safe(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return {"tensor_shape": list(obj.shape), "dtype": str(obj.dtype)}
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return str(obj)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def fail(audit: dict[str, Any], reason: str, out_dir: Path) -> None:
    audit["can_run_analysis"] = False
    audit.setdefault("missing_or_invalid", []).append(reason)
    write_json(out_dir / "artifact_audit.json", audit)
    print(f"[STOP] {reason}")
    print(f"Audit written to {out_dir / 'artifact_audit.json'}")
    raise SystemExit(2)


def load_feature_cache(path: str, audit: dict[str, Any], out_dir: Path):
    rec = file_record(path)
    audit["feature_cache"] = rec
    if not rec["exists"]:
        fail(audit, "feature cache is missing", out_dir)
    try:
        data = np.load(path, allow_pickle=True)
    except Exception as exc:
        fail(audit, f"failed to load feature cache: {exc}", out_dir)
    keys = list(data.keys())
    audit["feature_cache"]["keys"] = keys
    required = ["train_features", "train_labels"]
    for key in required:
        if key not in data:
            fail(audit, f"feature cache missing key {key}", out_dir)
    feats = data["train_features"]
    labels = data["train_labels"]
    audit["feature_cache"]["train_features_shape"] = list(feats.shape)
    audit["feature_cache"]["train_labels_shape"] = list(labels.shape)
    if feats.ndim != 2 or labels.ndim != 1 or feats.shape[0] != labels.shape[0]:
        fail(audit, "feature cache train feature/label shapes are invalid", out_dir)
    return feats.astype(np.float32), labels.astype(np.int64), keys


def state_dict_looks_like_mlp_refiner(sd: dict[str, Any]) -> bool:
    keys = set(sd.keys())
    return "class_embed.weight" in keys and any(k.startswith("net.") for k in keys)


def load_artifacts(args: argparse.Namespace, audit: dict[str, Any], out_dir: Path):
    payload = None
    refiner_state = None
    prototype_pool = None
    artifact_args = {}

    if args.artifact:
        audit["combined_artifact"] = file_record(args.artifact)
        if not audit["combined_artifact"]["exists"]:
            fail(audit, "combined analysis artifact is missing", out_dir)
        try:
            payload = torch.load(args.artifact, map_location="cpu")
        except Exception as exc:
            fail(audit, f"failed to load combined analysis artifact: {exc}", out_dir)
        if not isinstance(payload, dict):
            fail(audit, "combined analysis artifact is not a dict", out_dir)
        audit["combined_artifact"]["format"] = payload.get("format")
        audit["combined_artifact"]["keys"] = list(payload.keys())
        refiner_state = payload.get("refiner_state_dict")
        prototype_pool = payload.get("prototype_pool")
        artifact_args = payload.get("args") or {}
    else:
        audit["refiner_checkpoint"] = file_record(args.refiner_checkpoint)
        audit["prototype_pool_file"] = file_record(args.prototype_pool)
        if not args.refiner_checkpoint or not audit["refiner_checkpoint"]["exists"]:
            fail(audit, "final MLP refiner checkpoint is missing", out_dir)
        if not args.prototype_pool or not audit["prototype_pool_file"]["exists"]:
            fail(audit, "merged prototype pool file is missing", out_dir)
        try:
            refiner_state = torch.load(args.refiner_checkpoint, map_location="cpu")
        except Exception as exc:
            fail(audit, f"failed to load refiner checkpoint: {exc}", out_dir)
        try:
            prototype_payload = torch.load(args.prototype_pool, map_location="cpu")
        except Exception as exc:
            fail(audit, f"failed to load prototype pool: {exc}", out_dir)
        prototype_pool = prototype_payload.get("prototype_pool", prototype_payload) if isinstance(prototype_payload, dict) else prototype_payload

    if not isinstance(refiner_state, dict):
        fail(audit, "refiner state is missing or not a state_dict", out_dir)
    audit["refiner_state_keys"] = list(refiner_state.keys())[:80]
    if not state_dict_looks_like_mlp_refiner(refiner_state):
        fail(audit, "provided checkpoint does not look like an MLP refiner state_dict", out_dir)

    if not isinstance(prototype_pool, dict):
        fail(audit, "prototype pool is missing or not a dict", out_dir)

    audit["artifact_args"] = artifact_args
    audit["prototype_pool_classes"] = sorted([int(c) for c in prototype_pool.keys()])
    return refiner_state, prototype_pool, artifact_args


def flatten_class_prototypes(prototype_pool: dict, c: int):
    if c not in prototype_pool and str(c) in prototype_pool:
        class_pool = prototype_pool[str(c)]
    else:
        class_pool = prototype_pool.get(c)
    if class_pool is None:
        return [], []

    anchors = []
    counts = []
    if isinstance(class_pool, dict):
        iterator = class_pool.values()
    else:
        iterator = [class_pool]

    for entries in iterator:
        if entries is None:
            continue
        for item in entries:
            if isinstance(item, dict):
                mu = item.get("mu")
                if mu is None:
                    mu = item.get("prototype")
                if mu is None:
                    mu = item.get("anchor")
                n = item.get("n")
                if n is None:
                    n = item.get("count")
                if n is None:
                    n = item.get("weight")
            else:
                mu, n = item[0], item[1]
            if mu is None or n is None:
                continue
            mu = torch.as_tensor(mu, dtype=torch.float32).flatten()
            anchors.append(F.normalize(mu, dim=0))
            counts.append(max(float(n), 0.0))
    return anchors, counts


def normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=1)


def avg_pairwise_cosine_distance(x: torch.Tensor) -> float:
    x = normalize_rows(x)
    n = x.shape[0]
    if n < 2:
        return float("nan")
    sim = x @ x.T
    idx = torch.triu_indices(n, n, offset=1)
    return float((1.0 - sim[idx[0], idx[1]]).mean().item())


def effective_rank(x: torch.Tensor) -> float:
    xc = x.float() - x.float().mean(dim=0, keepdim=True)
    try:
        s = torch.linalg.svdvals(xc)
    except Exception:
        return float("nan")
    total = s.sum()
    if float(total.item()) <= 1e-12:
        return 0.0
    p = (s / total).clamp_min(1e-12)
    entropy = -(p * p.log()).sum()
    return float(torch.exp(entropy).item())


def mean_dist_to_centroid(x: torch.Tensor) -> float:
    centroid = x.float().mean(dim=0, keepdim=True)
    return float(torch.norm(x.float() - centroid, dim=1).mean().item())


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {"num_classes": len(rows)}
    keys = [k for k, v in rows[0].items() if isinstance(v, (int, float)) and k != "class"] if rows else []
    for k in keys:
        vals = np.array([r[k] for r in rows if isinstance(r.get(k), (int, float)) and math.isfinite(float(r[k]))], dtype=np.float64)
        if vals.size == 0:
            continue
        out[f"{k}_mean"] = float(vals.mean())
        out[f"{k}_std"] = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
    return out


def run_analysis(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audit: dict[str, Any] = {
        "analysis": "analysis1_synthetic_feature_distribution",
        "requested_setting": {
            "dataset": args.dataset,
            "alpha": args.alpha,
            "seed": args.seed,
            "partition_seed": args.partition_seed,
            "sigma": args.sigma,
            "samples_per_class": args.samples_per_class,
            "analysis_seed": args.analysis_seed,
        },
        "can_run_analysis": None,
        "missing_or_invalid": [],
    }

    if args.dataset != "cifar100" or abs(args.alpha - 0.01) > 1e-12 or args.seed != 42:
        fail(audit, "requested setting is not CIFAR-100 alpha=0.01 seed=42", out_dir)

    train_features_np, train_labels_np, _ = load_feature_cache(args.feature_cache, audit, out_dir)
    refiner_state, prototype_pool, artifact_args = load_artifacts(args, audit, out_dir)

    class_ids = sorted(set(train_labels_np.tolist()))
    audit["feature_cache"]["num_train_classes"] = len(class_ids)
    if class_ids != list(range(args.num_classes)):
        fail(audit, "feature cache does not cover all 100 CIFAR-100 classes", out_dir)

    missing_proto_classes = []
    proto_class_counts = {}
    for c in range(args.num_classes):
        anchors, counts = flatten_class_prototypes(prototype_pool, c)
        proto_class_counts[str(c)] = len(anchors)
        if len(anchors) == 0 or sum(counts) <= 0:
            missing_proto_classes.append(c)
    audit["prototype_pool_class_anchor_counts"] = proto_class_counts
    if missing_proto_classes:
        fail(audit, f"prototype pool missing usable anchors for classes: {missing_proto_classes[:20]}", out_dir)

    artifact_refiner_type = artifact_args.get("refiner_type") if isinstance(artifact_args, dict) else None
    if artifact_refiner_type not in (None, "mlp"):
        fail(audit, f"artifact refiner_type is not mlp: {artifact_refiner_type}", out_dir)

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    refiner = MLPRefiner(args.feat_dim, args.num_classes, args.refiner_hidden, args.refiner_layers)
    missing, unexpected = refiner.load_state_dict(refiner_state, strict=False)
    audit["refiner_load_state"] = {"missing": missing, "unexpected": unexpected}
    hard_missing = [k for k in missing if not k.startswith("proto_merge_threshold_learner")]
    if hard_missing:
        fail(audit, f"refiner state missing required keys: {hard_missing}", out_dir)
    refiner = refiner.to(device).eval()

    torch.manual_seed(args.analysis_seed)
    np.random.seed(args.analysis_seed)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(args.analysis_seed)

    train_features = normalize_rows(torch.from_numpy(train_features_np))
    train_labels = torch.from_numpy(train_labels_np)

    rows = []
    sign_improved = 0
    sign_worse = 0
    with torch.no_grad():
        for c in range(args.num_classes):
            real_c = train_features[train_labels == c]
            if real_c.shape[0] == 0:
                fail(audit, f"class {c} has no real training features", out_dir)

            anchors_list, counts_list = flatten_class_prototypes(prototype_pool, c)
            anchors = normalize_rows(torch.stack(anchors_list))
            counts = torch.tensor(counts_list, dtype=torch.float32)
            weights = counts / counts.sum()

            sampled_idx = torch.multinomial(weights, args.samples_per_class, replacement=True, generator=gen)
            sampled_mu = anchors[sampled_idx]
            eps = torch.randn(sampled_mu.shape, generator=gen, dtype=torch.float32)
            z0 = normalize_rows(sampled_mu + float(args.sigma) * eps)
            labels_c = torch.full((args.samples_per_class,), c, dtype=torch.long, device=device)
            zhat = normalize_rows(refiner.generate(z0.to(device), labels_c).cpu())

            real_centroid = F.normalize(real_c.mean(dim=0), dim=0)
            cos_real_naive = float((z0 @ real_centroid).mean().item())
            cos_real_refined = float((zhat @ real_centroid).mean().item())
            if cos_real_refined > cos_real_naive:
                sign_improved += 1
            elif cos_real_refined < cos_real_naive:
                sign_worse += 1

            nearest_proto_cos_naive = float((z0 @ anchors.T).max(dim=1).values.mean().item())
            nearest_proto_cos_refined = float((zhat @ anchors.T).max(dim=1).values.mean().item())
            dist_source_naive = float(torch.norm(z0 - sampled_mu, dim=1).mean().item())
            dist_source_refined = float(torch.norm(zhat - sampled_mu, dim=1).mean().item())

            if real_c.shape[0] >= 2:
                n_real = min(100, real_c.shape[0])
                real_idx = torch.randperm(real_c.shape[0], generator=gen)[:n_real]
                real_div = avg_pairwise_cosine_distance(real_c[real_idx])
            else:
                real_div = float("nan")

            div_naive = avg_pairwise_cosine_distance(z0)
            div_refined = avg_pairwise_cosine_distance(zhat)
            row = {
                "class": c,
                "num_real_features": int(real_c.shape[0]),
                "num_proto_anchors": int(anchors.shape[0]),
                "proto_count_sum": float(counts.sum().item()),
                "cos_to_real_centroid_naive": cos_real_naive,
                "cos_to_real_centroid_refined": cos_real_refined,
                "cos_to_real_centroid_delta": cos_real_refined - cos_real_naive,
                "nearest_proto_cos_naive": nearest_proto_cos_naive,
                "nearest_proto_cos_refined": nearest_proto_cos_refined,
                "source_anchor_l2_naive": dist_source_naive,
                "source_anchor_l2_refined": dist_source_refined,
                "pairwise_cos_dist_real": real_div,
                "pairwise_cos_dist_naive": div_naive,
                "pairwise_cos_dist_refined": div_refined,
                "diversity_ratio_naive_to_real": div_naive / real_div if math.isfinite(real_div) and real_div > 0 else float("nan"),
                "diversity_ratio_refined_to_real": div_refined / real_div if math.isfinite(real_div) and real_div > 0 else float("nan"),
                "effective_rank_naive": effective_rank(z0),
                "effective_rank_refined": effective_rank(zhat),
                "mean_dist_to_synth_centroid_naive": mean_dist_to_centroid(z0),
                "mean_dist_to_synth_centroid_refined": mean_dist_to_centroid(zhat),
            }
            rows.append(row)

    audit["can_run_analysis"] = True
    write_json(out_dir / "artifact_audit.json", audit)

    csv_path = out_dir / "per_class_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    summary["class_consistency_improved_classes"] = sign_improved
    summary["class_consistency_worse_classes"] = sign_worse
    summary["collapse_flag_nearest_proto_cos_refined_gt_0_995"] = (
        summary.get("nearest_proto_cos_refined_mean", 0.0) > 0.995
    )
    summary["supports_class_consistency"] = (
        summary.get("cos_to_real_centroid_refined_mean", -1.0)
        > summary.get("cos_to_real_centroid_naive_mean", 1.0)
    )
    summary["supports_non_collapse"] = not summary["collapse_flag_nearest_proto_cos_refined_gt_0_995"]
    summary["supports_diversity"] = summary.get("pairwise_cos_dist_refined_mean", 0.0) > 1e-6
    summary["supports_anti_collapse_claim"] = all([
        summary["supports_class_consistency"],
        summary["supports_non_collapse"],
        summary["supports_diversity"],
    ])
    write_json(out_dir / "summary_metrics.json", summary)

    metadata = {
        "dataset": args.dataset,
        "alpha": args.alpha,
        "seed": args.seed,
        "partition_seed": args.partition_seed,
        "sigma": args.sigma,
        "samples_per_class": args.samples_per_class,
        "analysis_seed": args.analysis_seed,
        "paired_noise": True,
        "anchor_sampling": "count_weighted_over_same_class_prototype_pool",
        "classes": args.num_classes,
    }
    write_json(out_dir / "generation_metadata.json", metadata)

    tests = {
        "paired_class_consistency_sign_test": {
            "improved": sign_improved,
            "worse": sign_worse,
            "ties": args.num_classes - sign_improved - sign_worse,
        }
    }
    write_json(out_dir / "paired_metric_tests.json", tests)

    print(f"Wrote {csv_path}")
    print(f"Wrote {out_dir / 'summary_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_analysis(parse_args()))
