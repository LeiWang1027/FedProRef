"""
FedProRef: Classification Head
- Linear head
- 1-hidden-layer MLP head
Backbone is frozen; only head parameters are trained.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class LinearHead(nn.Module):
    def __init__(self, feat_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        return self.fc(x)  # raw logits


class MLPHead(nn.Module):
    def __init__(self, feat_dim: int, num_classes: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def create_head(head_type: str, feat_dim: int, num_classes: int, hidden_dim: int = 512):
    if head_type == "linear":
        return LinearHead(feat_dim, num_classes)
    elif head_type == "mlp":
        return MLPHead(feat_dim, num_classes, hidden_dim)
    else:
        raise ValueError(f"Unknown head type: {head_type}")


def clone_head(head: nn.Module) -> nn.Module:
    return copy.deepcopy(head)


def fedavg_heads(global_head: nn.Module, client_heads: list,
                 weights: list = None) -> nn.Module:
    """Federated averaging of head parameters."""
    n = len(client_heads)
    if weights is None:
        weights = [1.0 / n] * n

    global_dict = global_head.state_dict()
    for key in global_dict:
        global_dict[key] = sum(
            w * client_heads[i].state_dict()[key].float()
            for i, w in enumerate(weights)
        )
    global_head.load_state_dict(global_dict)
    return global_head


def fuse_head(local_head: nn.Module, cal_head: nn.Module, beta: float) -> nn.Module:
    """Fuse calibrated head into local head: W = (1-beta)*W_local + beta*W_cal."""
    local_dict = local_head.state_dict()
    cal_dict = cal_head.state_dict()
    for key in local_dict:
        local_dict[key] = (1 - beta) * local_dict[key] + beta * cal_dict[key]
    local_head.load_state_dict(local_dict)
    return local_head
