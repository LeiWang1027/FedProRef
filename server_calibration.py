"""
Server-side calibration
1. Aggregate prototype statistics
2. Train refiner (MLP or RF) with prototype loss
3. Generate candidate features
4. Budget-constrained head calibration
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from proposal import sample_proposal
from refiner_mlp import MLPRefiner
from refiner_rf import RectifiedFlowRefiner
from head import clone_head


class LearnableProtoMergeThreshold(nn.Module):
    """Bounded scalar threshold used for prototype similarity merging."""

    def __init__(self, initial: float, min_value: float, max_value: float):
        super().__init__()
        if max_value <= min_value:
            raise ValueError("proto merge threshold max must be larger than min")
        self.min_value = float(min_value)
        self.max_value = float(max_value)

        initial = float(np.clip(initial, self.min_value + 1e-6, self.max_value - 1e-6))
        ratio = (initial - self.min_value) / (self.max_value - self.min_value)
        raw = np.log(ratio / (1.0 - ratio))
        self.raw_threshold = nn.Parameter(torch.tensor(raw, dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        span = self.max_value - self.min_value
        return self.min_value + span * torch.sigmoid(self.raw_threshold)

    def item(self) -> float:
        return float(self().detach().cpu().item())


def attach_proto_merge_threshold(refiner, args):
    if refiner is None:
        return refiner
    if hasattr(refiner, "proto_merge_threshold_learner"):
        return refiner

    refiner.proto_merge_threshold_learner = LearnableProtoMergeThreshold(
        args.proto_merge_threshold,
        args.proto_merge_threshold_min,
        args.proto_merge_threshold_max,
    )
    return refiner


def get_proto_merge_threshold(refiner, fallback: float) -> float:
    learner = getattr(refiner, "proto_merge_threshold_learner", None)
    if learner is None:
        return float(fallback)
    return learner.item()


def _collect_same_class_proto_sims(all_client_stats_list: list, num_classes: int, device: str):
    sims = []
    for c in range(num_classes):
        mus_c = []
        for stats in all_client_stats_list:
            if c not in stats["counts"]:
                continue
            for m in stats["counts"][c]:
                mus_c.append(stats["prototypes"][c][m])

        if len(mus_c) < 2:
            continue

        mus = F.normalize(torch.stack(mus_c).to(device), dim=1)
        sim_matrix = mus @ mus.T
        pair_idx = torch.triu_indices(len(mus_c), len(mus_c), offset=1, device=device)
        sims.append(sim_matrix[pair_idx[0], pair_idx[1]])

    if len(sims) == 0:
        return None
    return torch.cat(sims, dim=0)


def learn_proto_merge_threshold(
    refiner,
    all_client_stats_list: list,
    args,
    current_threshold: float,
    current_acc: float | None = None,
    best_acc: float | None = None,
    prev_update_acc: float | None = None,
):
    """Update the bounded merge threshold from refiner and global-accuracy feedback.

    Lower refiner prototype loss only permits a lower threshold when the global
    model accuracy is also near the historical best or improving from the last
    threshold update. This prevents prototype merging from being driven by a
    local refiner objective alone.
    """
    if refiner is None:
        return float(current_threshold), False, None
    if not getattr(args, "proto_similarity_merge", True):
        return float(current_threshold), False, None
    if not getattr(args, "proto_merge_adaptive", True):
        return float(current_threshold), False, None
    if not getattr(args, "proto_merge_learnable", True):
        return float(current_threshold), False, None

    refiner = attach_proto_merge_threshold(refiner, args)
    learner = getattr(refiner, "proto_merge_threshold_learner", None)
    if learner is None:
        return float(current_threshold), False, None

    metrics = getattr(refiner, "last_train_metrics", {}) if refiner is not None else {}
    proto_loss = metrics.get("proto")
    if proto_loss is None or current_acc is None:
        threshold = learner.item()
        return threshold, abs(threshold - current_threshold) > 1e-12, None

    sims = _collect_same_class_proto_sims(
        all_client_stats_list, args.num_classes, args.device)
    if sims is None or sims.numel() == 0:
        threshold = learner.item()
        return threshold, abs(threshold - current_threshold) > 1e-12, None

    low = float(args.proto_merge_target_low)
    high = float(args.proto_merge_target_high)
    if high <= low:
        proto_quality = 0.5
    else:
        proto_quality = (high - float(proto_loss)) / (high - low)
        proto_quality = float(np.clip(proto_quality, 0.0, 1.0))

    current_acc = float(current_acc)
    if best_acc is None:
        best_quality = 1.0
    else:
        best_gap = max(0.0, float(best_acc) - current_acc)
        best_quality = 1.0 - best_gap / max(float(args.proto_merge_acc_drop_tolerance), 1e-6)
        best_quality = float(np.clip(best_quality, 0.0, 1.0))

    if prev_update_acc is None:
        trend_quality = 0.5
        acc_delta = 0.0
    else:
        acc_delta = current_acc - float(prev_update_acc)
        scaled_delta = acc_delta / max(float(args.proto_merge_acc_gain_tolerance), 1e-6)
        trend_quality = 0.5 + 0.5 * float(np.clip(scaled_delta, -1.0, 1.0))

    acc_quality = 0.7 * best_quality + 0.3 * trend_quality
    acc_weight = float(np.clip(args.proto_merge_acc_weight, 0.0, 1.0))
    quality = (1.0 - acc_weight) * proto_quality + acc_weight * acc_quality
    quality = float(np.clip(quality, 0.0, 1.0))

    min_tau = float(args.proto_merge_threshold_min)
    max_tau = float(args.proto_merge_threshold_max)
    target_tau = max_tau - quality * (max_tau - min_tau)
    temp = max(float(args.proto_merge_temperature), 1e-4)
    steps = max(int(args.proto_merge_learn_steps), 1)

    learner = learner.to(args.device)
    optimizer = optim.Adam(learner.parameters(), lr=float(args.proto_merge_threshold_lr))
    before = learner.item()
    last_loss = None

    target_probs = torch.sigmoid((sims.detach() - target_tau) / temp).detach()
    for _ in range(steps):
        threshold = learner()
        merge_probs = torch.sigmoid((sims.detach() - threshold) / temp)
        bce = F.binary_cross_entropy(
            merge_probs.clamp(1e-6, 1.0 - 1e-6),
            target_probs.clamp(1e-6, 1.0 - 1e-6),
        )
        tau_loss = F.mse_loss(threshold, torch.tensor(target_tau, device=args.device))
        loss = bce + float(args.proto_merge_tau_reg) * tau_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu().item())

    after = learner.item()
    info = {
        "loss": last_loss,
        "proto_loss": float(proto_loss),
        "current_acc": current_acc,
        "best_acc": None if best_acc is None else float(best_acc),
        "prev_update_acc": None if prev_update_acc is None else float(prev_update_acc),
        "acc_delta": float(acc_delta),
        "proto_quality": float(proto_quality),
        "acc_quality": float(acc_quality),
        "quality": float(quality),
        "target_tau": float(target_tau),
        "pairs": int(sims.numel()),
    }
    return after, abs(after - before) > 1e-6, info


def _prototype_mass(prototypes_and_counts: list) -> int:
    return sum(max(int(n), 0) for _, n in prototypes_and_counts)


def _allocate_mode_samples(class_prototypes: dict, total_samples: int):
    active_modes = [
        (m, class_prototypes[m])
        for m in class_prototypes
        if len(class_prototypes[m]) > 0
    ]
    if total_samples <= 0 or len(active_modes) == 0:
        return {}

    masses = torch.tensor(
        [_prototype_mass(protos) for _, protos in active_modes],
        dtype=torch.float32,
    )
    if masses.sum() < 1e-8:
        masses = torch.ones_like(masses)

    if total_samples < len(active_modes):
        order = torch.argsort(masses, descending=True).tolist()
        selected = set(order[:total_samples])
        return {m: (1 if i in selected else 0) for i, (m, _) in enumerate(active_modes)}

    raw = masses / masses.sum() * total_samples
    alloc = torch.floor(raw).long().clamp_min(1)
    remaining = total_samples - int(alloc.sum().item())
    fractions = raw - torch.floor(raw)

    if remaining > 0:
        order = torch.argsort(fractions, descending=True).tolist()
        for i in range(remaining):
            alloc[order[i % len(order)]] += 1
    elif remaining < 0:
        order = torch.argsort(fractions, descending=False).tolist()
        need = -remaining
        i = 0
        while need > 0 and i < len(order) * total_samples:
            idx = order[i % len(order)]
            if alloc[idx] > 1:
                alloc[idx] -= 1
                need -= 1
            i += 1

    return {m: int(alloc[i].item()) for i, (m, _) in enumerate(active_modes)}



def _nearest_proto_targets(z0: torch.Tensor, proto_mus: torch.Tensor) -> torch.Tensor:
    dists = torch.cdist(z0.detach(), proto_mus)
    nearest = dists.argmin(dim=1)
    return proto_mus[nearest].detach()


def train_refiner(
    refiner,
    all_prototypes: dict,
    num_classes: int,
    gen_per_class: int,
    proposal_sigma: float,
    refiner_epochs: int,
    refiner_lr: float,
    w_reg: float,
    w_proto: float,
    w_flow: float,
    device: str,
    refiner_type: str = "mlp",
):
    # Each class keeps a fixed sampling budget, split across its global
    # prototype clusters by cluster mass. The optimizer steps once per class,
    # so classes with more clusters do not receive more update steps.
    refiner = refiner.to(device)
    optimizer = optim.Adam(refiner.parameters(), lr=refiner_lr)

    active_classes = []
    for c in range(num_classes):
        if c not in all_prototypes:
            continue
        allocations = {
            m: n_samples
            for m, n_samples in _allocate_mode_samples(all_prototypes[c], gen_per_class).items()
            if n_samples > 0
        }
        if len(allocations) > 0:
            active_classes.append((c, allocations))

    if len(active_classes) == 0:
        print("  [Server] No active classes for refiner training")
        refiner.last_train_metrics = {"loss": None, "reg": None, "proto": None, "flow": None}
        return refiner

    last_metrics = {"loss": None, "reg": None, "proto": None, "flow": None}

    for epoch in range(refiner_epochs):
        total_loss = 0.0
        total_reg_loss = 0.0
        total_proto_loss = 0.0
        total_flow_loss = 0.0

        for c, allocations in active_classes:
            class_total = sum(allocations.values())
            class_loss = None
            class_reg_loss = 0.0
            class_proto_loss = 0.0
            class_flow_loss = 0.0

            optimizer.zero_grad()
            for m, n_samples in allocations.items():
                z0 = sample_proposal(all_prototypes[c][m], n_samples,
                                     sigma=proposal_sigma, device=device)
                labels_c = torch.full((n_samples,), c, dtype=torch.long, device=device)

                protos_cm = all_prototypes[c][m]
                proto_mus = torch.stack([mu for mu, n in protos_cm]).to(device)

                if refiner_type == "rf":
                    z_target = _nearest_proto_targets(z0, proto_mus)
                    L_flow = refiner.flow_matching_loss(z0, z_target, labels_c)
                    z_gen = refiner.generate(z0, labels_c)
                else:
                    L_flow = torch.tensor(0.0, device=device)
                    z_gen = refiner(z0, labels_c)

                if refiner_type == "rf":
                    t_rand = torch.rand(n_samples, device=device)
                    v = refiner(z0, t_rand, labels_c)
                    L_reg = (v ** 2).mean()
                else:
                    delta = z_gen - z0
                    L_reg = (delta ** 2).mean()

                dists = torch.cdist(z_gen, proto_mus)
                L_proto = dists.min(dim=1).values.mean()

                mode_weight = n_samples / class_total
                mode_loss = w_reg * L_reg + w_proto * L_proto + w_flow * L_flow
                weighted_loss = mode_weight * mode_loss
                class_loss = weighted_loss if class_loss is None else class_loss + weighted_loss
                class_reg_loss += mode_weight * L_reg.item()
                class_proto_loss += mode_weight * L_proto.item()
                class_flow_loss += mode_weight * L_flow.item()

            if class_loss is None:
                continue
            class_loss.backward()
            optimizer.step()

            total_loss += class_loss.item()
            total_reg_loss += class_reg_loss
            total_proto_loss += class_proto_loss
            total_flow_loss += class_flow_loss

        nc = len(active_classes)
        last_metrics = {
            "loss": total_loss / nc,
            "reg": total_reg_loss / nc,
            "proto": total_proto_loss / nc,
            "flow": total_flow_loss / nc,
        }
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"    Refiner epoch {epoch+1}/{refiner_epochs}: "
                  f"loss={last_metrics['loss']:.4f} "
                  f"flow={last_metrics['flow']:.4f} "
                  f"reg={last_metrics['reg']:.4f} "
                  f"proto={last_metrics['proto']:.4f}")

    refiner.last_train_metrics = last_metrics
    return refiner

def generate_calibration_features(
    refiner,
    all_prototypes: dict,
    num_classes: int,
    gen_per_class: int,
    proposal_sigma: float,
    device: str,
    use_refiner: bool = True,
) -> tuple:
    """
    Generate candidate features for calibration (per-mode sampling).

    Returns:
        features: (N_total, d) tensor
        labels: (N_total,) tensor
    """
    if refiner is not None:
        refiner.eval()
    all_feats = []
    all_labels = []

    for c in range(num_classes):
        if c not in all_prototypes:
            continue

        allocations = _allocate_mode_samples(all_prototypes[c], gen_per_class)
        for m, n_samples in allocations.items():
            if n_samples <= 0:
                continue

            z0 = sample_proposal(all_prototypes[c][m], n_samples,
                                 sigma=proposal_sigma, device=device)
            labels_c = torch.full((n_samples,), c, dtype=torch.long, device=device)

            if use_refiner and refiner is not None:
                with torch.no_grad():
                    z = refiner.generate(z0, labels_c)
            else:
                z = z0

            all_feats.append(z.cpu())
            all_labels.append(labels_c.cpu())

    if len(all_feats) == 0:
        return torch.empty(0, 512), torch.empty(0, dtype=torch.long)

    features = torch.cat(all_feats, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return features, labels


def budget_select(features: torch.Tensor, labels: torch.Tensor,
                  budget: int, num_classes: int) -> tuple:
    """
    Select a budget-constrained subset for calibration.
    Strategy: uniform per-class allocation.

    Returns:
        sel_features: (B, d)
        sel_labels: (B,)
    """
    per_class = max(1, budget // num_classes)
    sel_feats = []
    sel_labs = []

    for c in range(num_classes):
        mask = labels == c
        feats_c = features[mask]
        if len(feats_c) == 0:
            continue
        n_sel = min(per_class, len(feats_c))
        # Random subset
        idx = torch.randperm(len(feats_c))[:n_sel]
        sel_feats.append(feats_c[idx])
        sel_labs.append(torch.full((n_sel,), c, dtype=torch.long))

    return torch.cat(sel_feats), torch.cat(sel_labs)


def calibrate_head(head, features: torch.Tensor, labels: torch.Tensor,
                   cal_epochs: int, cal_lr: float, device: str):
    """
    Fine-tune a copy of the head using synthetic calibration features.

    Returns:
        cal_head: calibrated head model
    """
    cal_head = clone_head(head).to(device)
    optimizer = optim.Adam(cal_head.parameters(), lr=cal_lr)
    criterion = nn.CrossEntropyLoss()

    features = features.to(device)
    labels = labels.to(device)

    cal_head.train()
    for epoch in range(cal_epochs):
        logits = cal_head(features)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return cal_head


def generate_weak_class_features(
    refiner,
    client_labels: np.ndarray,
    global_prototypes: dict,    # {c: {m: list of (mu, n)}}  — 服务端汇聚的全局原型（按mode）
    min_samples: int,
    num_classes: int,
    aug_gen_per_class: int,
    proposal_sigma: float,
    device: str,
    weak_percentile: float = 0.0,
    use_refiner: bool = True,   # False → 仅高斯采样，不使用 refiner（proto_aug 用）
    feature_generator=None,
) -> tuple:
    """
    为客户端的弱类/缺失类生成合成特征，用于本地训练数据增强。

    弱类判定逻辑（满足任一条件即合成）：
      条件1 — 绝对阈值: n_c < min_samples
      条件2 — 百分位阈值: n_c 排在所有类中样本数倒数 weak_percentile% 以内
                          例如 weak_percentile=20 → 样本数最少的 20% 的类

    每个弱类固定生成 aug_gen_per_class 个合成特征。

    只对有全局原型可用的弱类/缺失类进行合成。
    Returns: (aug_feats, aug_labels) — 均在 CPU 上的 Tensor
    """
    per_class_count = np.bincount(client_labels.astype(int), minlength=num_classes)

    # 百分位阈值：倒数 weak_percentile% 对应的样本数上界
    if weak_percentile > 0.0:
        pct_threshold = float(np.percentile(per_class_count, weak_percentile))
    else:
        pct_threshold = -1.0  # 不启用百分位条件

    def is_weak(c):
        n = int(per_class_count[c])
        return (n < min_samples) or (pct_threshold >= 0 and n <= pct_threshold)

    # 筛选出弱类/缺失类（且服务端有对应原型）
    weak_protos = {}

    for c in range(num_classes):
        if not is_weak(c):
            continue
        if c not in global_prototypes:
            continue

        # Collect all modes for this weak class
        weak_protos[c] = {}
        for m in global_prototypes[c]:
            if len(global_prototypes[c][m]) > 0:
                weak_protos[c][m] = global_prototypes[c][m]

    if len(weak_protos) == 0:
        return torch.empty(0, dtype=torch.float32), torch.empty(0, dtype=torch.long)

    if feature_generator is None:
        # 每个弱类总计生成 aug_gen_per_class 个样本，按全局原型簇分摊。
        aug_feats, aug_labels = generate_calibration_features(
            refiner if use_refiner else None, weak_protos, num_classes,
            aug_gen_per_class, proposal_sigma, device, use_refiner=use_refiner)
    else:
        aug_feats, aug_labels = feature_generator(weak_protos)

    return aug_feats, aug_labels  # already on CPU


def create_refiner(args):
    """Create refiner model based on args.refiner_type."""
    if args.refiner_type == "mlp":
        refiner = MLPRefiner(args.feat_dim, args.num_classes,
                             args.refiner_hidden, args.refiner_layers)
    elif args.refiner_type == "rf":
        refiner = RectifiedFlowRefiner(args.feat_dim, args.num_classes,
                                       args.refiner_hidden, args.refiner_layers,
                                       args.rf_steps)
    else:
        return None
    return attach_proto_merge_threshold(refiner, args)


def client_calibrate_with_refiner(
    refiner,
    client_prototypes: dict,  # {c: {m: mu}}
    client_counts: dict,      # {c: {m: n}}
    head,
    args,
    device: str,
):
    """
    Client-side calibration using the globally broadcast refiner.
    Each client generates synthetic features from its OWN prototypes,
    then calibrates a local copy of the head.
    Returns: calibrated head (clone).
    """
    # Convert to all_prototypes format: {c: {m: list of (mu, n)}}
    local_protos = {}
    for c in client_prototypes:
        modes = client_prototypes[c]
        if len(modes) == 0:
            continue
        local_protos[c] = {}
        for m in modes:
            local_protos[c][m] = [(modes[m], client_counts[c][m])]

    if len(local_protos) == 0:
        return clone_head(head)

    gen_feats, gen_labels = generate_calibration_features(
        refiner, local_protos, args.num_classes,
        args.gen_per_class, args.proposal_sigma, device, use_refiner=True)

    sel_feats, sel_labels = budget_select(
        gen_feats, gen_labels, args.cal_budget, args.num_classes)

    return calibrate_head(
        head, sel_feats, sel_labels, args.cal_epochs, args.cal_lr, device)


def server_calibration_pipeline(
    global_head,
    all_prototypes: dict,
    args,
    refiner=None,
    should_train_refiner: bool = True,
):
    """
    Full server-side calibration pipeline.

    1. Create/reuse refiner
    2. Train refiner
    3. Generate candidate features
    4. Budget-select
    5. Calibrate head
    6. Return calibrated head and refiner

    Returns:
        cal_head, refiner
    """
    device = args.device

    # ── Step 1: Create refiner if needed ──
    if refiner is None:
        if args.refiner_type == "mlp":
            refiner = MLPRefiner(args.feat_dim, args.num_classes,
                                 args.refiner_hidden, args.refiner_layers)
        elif args.refiner_type == "rf":
            refiner = RectifiedFlowRefiner(args.feat_dim, args.num_classes,
                                           args.refiner_hidden, args.refiner_layers,
                                           args.rf_steps)
        else:
            refiner = None

    use_refiner = refiner is not None and args.refiner_type != "none"

    # ── Step 2: Train refiner ──
    if use_refiner and should_train_refiner:
        print("  [Server] Training refiner...")
        refiner = train_refiner(
            refiner=refiner,
            all_prototypes=all_prototypes,
            num_classes=args.num_classes,
            gen_per_class=args.gen_per_class,
            proposal_sigma=args.proposal_sigma,
            refiner_epochs=getattr(args, "refiner_epochs", args.refiner_finetune_epochs),
            refiner_lr=args.refiner_lr,
            w_reg=args.w_reg,
            w_proto=args.w_proto,
            w_flow=args.w_flow,
            device=device,
            refiner_type=args.refiner_type,
        )

    # ── Step 3: Generate calibration features ──
    print("  [Server] Generating calibration features...")
    if use_refiner:
        gen_feats, gen_labels = generate_calibration_features(
            refiner, all_prototypes, args.num_classes,
            args.gen_per_class, args.proposal_sigma, device, use_refiner=True)
    else:
        # No refiner — just use proposal samples
        gen_feats, gen_labels = generate_calibration_features(
            None, all_prototypes, args.num_classes,
            args.gen_per_class, args.proposal_sigma, device, use_refiner=False)

    # ── Step 4: Budget select ──
    sel_feats, sel_labels = budget_select(
        gen_feats, gen_labels, args.cal_budget, args.num_classes)
    print(f"  [Server] Selected {len(sel_feats)} samples (budget={args.cal_budget})")

    # ── Step 5: Calibrate head ──
    print("  [Server] Calibrating head...")
    cal_head = calibrate_head(
        global_head, sel_feats, sel_labels,
        args.cal_epochs, args.cal_lr, device)

    return cal_head, refiner
