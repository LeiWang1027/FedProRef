"""DirectAnchorAug: direct sampled-anchor synthetic feature generation."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from proposal import sample_count_weighted_anchor_indices
from server_calibration import _allocate_mode_samples, generate_weak_class_features


ANCHOR_RNG_SCHEME = "sha256(training_seed,round_id,client_id,class_id,mode_id,direct_anchor_sampling)"


def stable_direct_anchor_seed(training_seed: int, round_id: int, client_id: int,
                              class_id: int, mode_id: int = 0) -> int:
    payload = "|".join(map(str, (
        training_seed, round_id, client_id, class_id, mode_id,
        "direct_anchor_sampling"))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def make_direct_anchor_generator(*, training_seed: int, round_id: int,
                                 client_id: int, class_id: int, mode_id: int,
                                 device: torch.device) -> torch.Generator:
    generator = torch.Generator(device=device.type)
    generator.manual_seed(stable_direct_anchor_seed(
        training_seed, round_id, client_id, class_id, mode_id))
    return generator


def generate_direct_anchor_features(*, anchor_vectors: torch.Tensor,
                                    anchor_counts: torch.Tensor,
                                    num_samples: int,
                                    generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw count-weighted active anchors and return detached normalized copies."""
    if anchor_vectors.ndim != 2:
        raise ValueError("anchor_vectors must have shape [anchors, feature_dim]")
    if anchor_vectors.shape[0] != anchor_counts.numel():
        raise ValueError("anchor_vectors and anchor_counts must have matching length")
    indices = sample_count_weighted_anchor_indices(
        anchor_counts.to(anchor_vectors.device), num_samples, generator)
    selected = anchor_vectors[indices].detach().clone()
    selected = F.normalize(selected, dim=-1).detach()
    selected.requires_grad_(False)
    return selected, indices


def _mode_direct_features(class_prototypes: dict, class_id: int, num_samples: int,
                          training_seed: int, round_id: int, client_id: int,
                          device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    features = []
    labels = []
    for mode_id, allocated in _allocate_mode_samples(class_prototypes, num_samples).items():
        if allocated <= 0:
            continue
        anchors = class_prototypes[mode_id]
        if not anchors:
            continue
        vectors = torch.stack([mu for mu, _ in anchors]).to(device)
        counts = torch.tensor([count for _, count in anchors], dtype=torch.float32, device=device)
        generator = make_direct_anchor_generator(
            training_seed=training_seed, round_id=round_id, client_id=client_id,
            class_id=class_id, mode_id=int(mode_id), device=vectors.device)
        generated, _ = generate_direct_anchor_features(
            anchor_vectors=vectors, anchor_counts=counts, num_samples=allocated,
            generator=generator)
        features.append(generated.cpu())
        labels.append(torch.full((allocated,), class_id, dtype=torch.long))
    if not features:
        return torch.empty((0, 0), dtype=torch.float32), torch.empty(0, dtype=torch.long)
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


@dataclass
class DirectAnchorAugmentation:
    """Method-local augmentation provider injected into the shared FL pipeline."""
    args: object
    cache: dict = field(default_factory=dict)
    pool_hash: str | None = None
    cache_hits: int = 0
    cache_misses: int = 0

    @staticmethod
    def _pool_hash(pool: dict) -> str:
        digest = hashlib.sha256()
        for class_id in sorted(pool):
            digest.update(str(class_id).encode())
            for mode_id in sorted(pool[class_id]):
                digest.update(str(mode_id).encode())
                for vector, count in pool[class_id][mode_id]:
                    tensor = vector.detach().cpu().contiguous()
                    digest.update(tensor.numpy().tobytes())
                    digest.update(str(int(count)).encode())
        return digest.hexdigest()

    def __call__(self, *, client_id: int, client_labels, global_prototypes: dict,
                 round_id: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        self.pool_hash = self._pool_hash(global_prototypes)
        target_device = torch.device(device)

        def generate(weak_prototypes: dict) -> tuple[torch.Tensor, torch.Tensor]:
            all_features = []
            all_labels = []
            for class_id in sorted(weak_prototypes):
                if not weak_prototypes[class_id]:
                    continue
                key = ("direct_anchor_aug", self.pool_hash, int(self.args.seed),
                       int(client_id), int(class_id), int(self.args.aug_gen_per_class))
                if key in self.cache:
                    self.cache_hits += 1
                    features, labels = self.cache[key]
                else:
                    self.cache_misses += 1
                    features, labels = _mode_direct_features(
                        weak_prototypes[class_id], int(class_id),
                        int(self.args.aug_gen_per_class), int(self.args.seed),
                        int(round_id), int(client_id), target_device)
                    self.cache[key] = (features, labels)
                if labels.numel():
                    all_features.append(features)
                    all_labels.append(labels)
            if not all_features:
                return torch.empty(0, dtype=torch.float32), torch.empty(0, dtype=torch.long)
            return torch.cat(all_features), torch.cat(all_labels)

        return generate_weak_class_features(
            None, client_labels, global_prototypes, self.args.min_samples_per_class,
            self.args.num_classes, self.args.aug_gen_per_class,
            self.args.proposal_sigma, device, weak_percentile=self.args.weak_class_percentile,
            use_refiner=False, feature_generator=generate)

    def metadata(self) -> dict:
        return {
            "method": "DirectAnchorAug",
            "method_slug": "direct_anchor_aug",
            "synthetic_mode": "direct_anchor",
            "uses_proposal_noise": False,
            "uses_refiner": False,
            "prototype_merge": bool(getattr(self.args, "proto_similarity_merge", True)),
            "num_aug_per_class": int(self.args.aug_gen_per_class),
            "fedavg_weight_includes_synthetic": False,
            "fedavg_weight_uses_original_real_count": True,
            "anchor_sampling_rule": "count_weighted_per_active_mode_with_existing_mode_allocation",
            "anchor_rng_scheme": ANCHOR_RNG_SCHEME,
            "pool_hash": self.pool_hash,
            "partition_seed": int(self.args.partition_seed),
            "training_seed": int(self.args.seed),
            "synthetic_cache": {"hits": self.cache_hits, "misses": self.cache_misses},
        }
