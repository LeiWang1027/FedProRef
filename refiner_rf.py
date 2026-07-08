"""
FedProRef: Rectified Flow feature refiner.
v_theta(z_t, t, c) is trained with flow matching and sampled by Euler integration.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmb(nn.Module):
    """Sinusoidal positional embedding for time t ∈ [0, 1]."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (N,) -> (N, dim)"""
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=1)


class VelocityField(nn.Module):
    """
    Velocity network v_theta(z, t, c).
    Input: concat(z, time_emb, class_emb)
    Output: velocity in feature space (d,)
    """
    def __init__(self, feat_dim: int, num_classes: int,
                 hidden_dim: int = 512, num_layers: int = 3):
        super().__init__()
        self.time_emb = SinusoidalTimeEmb(feat_dim)
        self.class_emb = nn.Embedding(num_classes, feat_dim)

        layers = []
        in_dim = feat_dim * 3  # z + time_emb + class_emb
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, feat_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, t: torch.Tensor,
                class_labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (N, d)
            t: (N,) time in [0, 1]
            class_labels: (N,) int

        Returns:
            v: (N, d) velocity
        """
        t_emb = self.time_emb(t)                  # (N, d)
        c_emb = self.class_emb(class_labels)       # (N, d)
        inp = torch.cat([z, t_emb, c_emb], dim=1)  # (N, 3d)
        return self.net(inp)


class RectifiedFlowRefiner(nn.Module):
    """
    Wraps VelocityField with flow-matching training and Euler integration.
    """
    def __init__(self, feat_dim: int, num_classes: int,
                 hidden_dim: int = 512, num_layers: int = 3, num_steps: int = 4):
        super().__init__()
        self.vf = VelocityField(feat_dim, num_classes, hidden_dim, num_layers)
        self.num_steps = num_steps

    def forward(self, z: torch.Tensor, t: torch.Tensor,
                class_labels: torch.Tensor) -> torch.Tensor:
        """Return velocity at (z, t, c)."""
        return self.vf(z, t, class_labels)

    def generate(self, z0: torch.Tensor, class_labels: torch.Tensor) -> torch.Tensor:
        """
        Euler integration from z0 (t=0) to z1 (t=1).

        Args:
            z0: (N, d) proposal samples
            class_labels: (N,)

        Returns:
            z: (N, d) refined, L2-normalized features
        """
        N = z0.shape[0]
        dt = 1.0 / self.num_steps
        z = z0.clone()

        for i in range(self.num_steps):
            t_val = i * dt
            t = torch.full((N,), t_val, device=z.device)
            v = self.vf(z, t, class_labels)
            z = z + dt * v
            z = F.normalize(z, dim=1)

        return z

    def flow_matching_loss(self, z0: torch.Tensor, z1: torch.Tensor,
                           class_labels: torch.Tensor) -> torch.Tensor:
        """Rectified-flow objective from proposal z0 to prototype target z1."""
        n = z0.shape[0]
        t = torch.rand(n, device=z0.device)
        z_t = (1.0 - t.unsqueeze(1)) * z0 + t.unsqueeze(1) * z1
        v_pred = self.vf(z_t, t, class_labels)
        v_target = z1 - z0
        return F.mse_loss(v_pred, v_target)
