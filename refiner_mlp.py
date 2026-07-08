"""
FedProRef: MLP Refiner
z = norm(z_0 + r_phi(z_0, c))
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPRefiner(nn.Module):
    """
    Residual MLP refiner.
    Takes z_0 (proposal) + class embedding -> outputs residual delta.
    """
    def __init__(self, feat_dim: int, num_classes: int,
                 hidden_dim: int = 512, num_layers: int = 3):
        super().__init__()
        self.class_embed = nn.Embedding(num_classes, feat_dim)

        layers = []
        in_dim = feat_dim * 2  # concat(z_0, class_embed)
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, feat_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z0: torch.Tensor, class_labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z0: (N, d) proposal features
            class_labels: (N,) integer class labels

        Returns:
            z: (N, d) refined, L2-normalized features
        """
        c_emb = self.class_embed(class_labels)  # (N, d)
        inp = torch.cat([z0, c_emb], dim=1)     # (N, 2d)
        delta = self.net(inp)                     # (N, d)
        z = z0 + delta
        z = F.normalize(z, dim=1)
        return z

    def generate(self, z0: torch.Tensor, class_labels: torch.Tensor) -> torch.Tensor:
        """Alias for forward (used at inference)."""
        return self.forward(z0, class_labels)
