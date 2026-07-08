"""
Client-side statistics
- Per-class feature collection
- K-means multi-mode decomposition
- Prototype computation
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans


PROTO_MERGE_SIM_THRESHOLD = 0.90


def compute_client_stats(features: np.ndarray, labels: np.ndarray,
                         num_classes: int, num_modes: int,
                         device: str = "cpu", min_samples: int = 1) -> dict:
    """
    Compute per-client upload statistics.

    Returns:
        {
            "prototypes": {c: {m: mu}},       # mu shape (d,)
            "counts":     {c: {m: n}},        # int
        }
    """
    prototypes = {}
    counts = {}

    for c in range(num_classes):
        mask = labels == c
        feats_c = features[mask]
        n_c = len(feats_c)

        prototypes[c] = {}
        counts[c] = {}

        if n_c == 0 or n_c < min_samples:
            continue

        K = min(num_modes, n_c)

        if K <= 1:
            mu = feats_c.mean(axis=0)
            mu = mu / (np.linalg.norm(mu) + 1e-8)
            prototypes[c][0] = torch.from_numpy(mu).float().to(device)
            counts[c][0] = n_c
        else:
            km = KMeans(n_clusters=K, n_init=3, max_iter=50, random_state=0)
            assignments = km.fit_predict(feats_c)
            for m in range(K):
                feats_m = feats_c[assignments == m]
                n_m = len(feats_m)
                if n_m == 0:
                    continue
                mu = feats_m.mean(axis=0)
                mu = mu / (np.linalg.norm(mu) + 1e-8)
                prototypes[c][m] = torch.from_numpy(mu).float().to(device)
                counts[c][m] = n_m

    return {
        "prototypes": prototypes,
        "counts": counts,
    }

def _connected_components(sim_matrix: torch.Tensor, threshold: float):
    n = sim_matrix.shape[0]
    seen = [False] * n
    components = []

    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        component = []

        while stack:
            i = stack.pop()
            component.append(i)
            neighbors = (sim_matrix[i] >= threshold).nonzero(as_tuple=False).flatten().tolist()
            for j in neighbors:
                if not seen[j]:
                    seen[j] = True
                    stack.append(j)

        components.append(component)

    return components


def _merge_prototype_cluster(contributions: list, device: str):
    if len(contributions) == 1:
        ct = contributions[0]
        return ct["mu"], ct["n"]

    total_n = sum(ct["n"] for ct in contributions)
    mus = F.normalize(torch.stack([ct["mu"] for ct in contributions]).to(device), dim=1)
    counts = torch.tensor([ct["n"] for ct in contributions], dtype=torch.float32, device=device)

    mu_ref = F.normalize((counts[:, None] * mus).sum(dim=0, keepdim=True), dim=1).squeeze(0)
    sims = F.cosine_similarity(mus, mu_ref.unsqueeze(0), dim=1).clamp_min(0.0)
    raw_weights = counts * sims

    if raw_weights.sum() < 1e-8:
        raw_weights = counts

    weights = raw_weights / raw_weights.sum()
    mu_global = F.normalize((weights[:, None] * mus).sum(dim=0, keepdim=True), dim=1).squeeze(0)
    return mu_global, total_n


def aggregate_client_stats(all_client_stats: list, num_classes: int, feat_dim: int,
                           num_modes: int, device: str = "cpu",
                           merge_sim_threshold: float = PROTO_MERGE_SIM_THRESHOLD,
                           use_similarity_merge: bool = True):
    # When similarity merge is enabled, build cosine-similarity components per
    # class. Otherwise, keep every uploaded prototype as its own global cluster.
    all_prototypes = {}

    for c in range(num_classes):
        all_prototypes[c] = {}
        contributions = []

        for stats in all_client_stats:
            if c not in stats["counts"]:
                continue
            for m in stats["counts"][c]:
                contributions.append({
                    "n": stats["counts"][c][m],
                    "mu": stats["prototypes"][c][m],
                })

        if len(contributions) == 0:
            continue

        if not use_similarity_merge:
            for cluster_id, ct in enumerate(contributions):
                all_prototypes[c][cluster_id] = [(ct["mu"], ct["n"])]
            continue

        if len(contributions) == 1:
            mu_global, total_n = _merge_prototype_cluster(contributions, device)
            all_prototypes[c][0] = [(mu_global, total_n)]
            continue

        mus = F.normalize(torch.stack([ct["mu"] for ct in contributions]).to(device), dim=1)
        sim_matrix = mus @ mus.T
        components = _connected_components(sim_matrix, merge_sim_threshold)

        for cluster_id, component in enumerate(components):
            cluster_contribs = [contributions[i] for i in component]
            mu_global, total_n = _merge_prototype_cluster(cluster_contribs, device)
            all_prototypes[c][cluster_id] = [(mu_global, total_n)]

    return all_prototypes
