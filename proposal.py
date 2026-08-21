"""
FedProRef: Proposal Distribution
Sample initial z_0 from prototype mixture + noise.
"""
import torch
import torch.nn.functional as F


def sample_count_weighted_anchor_indices(
        counts: torch.Tensor,
        num_samples: int,
        generator: torch.Generator | None = None) -> torch.Tensor:
    """Sample anchor indices with replacement in proportion to real counts."""
    if counts.ndim != 1 or counts.numel() == 0:
        raise ValueError("counts must be a nonempty one-dimensional tensor")
    if num_samples < 0:
        raise ValueError("num_samples must be nonnegative")
    weights = counts.to(dtype=torch.float32)
    if torch.any(weights < 0):
        raise ValueError("anchor counts must be nonnegative")
    if not torch.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("anchor counts must have finite positive mass")
    weights = weights / weights.sum()
    return torch.multinomial(
        weights, num_samples, replacement=True, generator=generator)


def sample_proposal(prototypes_and_counts: list, num_samples: int,
                    sigma: float = 0.05, device: str = "cpu") -> torch.Tensor:
    """
    Sample from prototype mixture proposal.

    Args:
        prototypes_and_counts: list of (mu, n) where mu is (d,) tensor, n is int
        num_samples: how many samples to draw
        sigma: noise scale
        device: target device

    Returns:
        z0: (num_samples, d) L2-normalized samples
    """
    if len(prototypes_and_counts) == 0:
        raise ValueError("No prototypes available for sampling")

    mus = torch.stack([p[0] for p in prototypes_and_counts]).to(device)  # (K, d)
    weights = torch.tensor([p[1] for p in prototypes_and_counts],
                           dtype=torch.float32, device=device)
    weights = weights / weights.sum()

    # Sample which prototype each sample comes from
    indices = torch.multinomial(weights, num_samples, replacement=True)  # (num_samples,)

    # Get base positions
    z0 = mus[indices]  # (num_samples, d)

    # Add noise
    noise = torch.randn_like(z0) * sigma
    z0 = z0 + noise

    # L2 normalize
    z0 = F.normalize(z0, dim=1)

    return z0
