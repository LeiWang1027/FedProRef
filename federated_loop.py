"""
FedProRef: Main Federated Learning Loop
Supports methods:
  - fedavg:       FedAvg baseline (head-only)
  - proto_aug:    Prototype Gaussian augmentation (no refiner, ablation)
  - proto_cal:    Prototype calibration (no sampling, no refiner, ablation)
  - proto_sample: Prototype mixture sampling + calibration (ablation)
  - fedproref:    Prototype-refiner method with weak-class augmentation
"""
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

from config import get_args
from utils import set_seed, evaluate_all, Logger
from data_utils import get_or_extract_features, FeatureDataset
from head import create_head, clone_head, fedavg_heads, fuse_head
from server_calibration import (generate_calibration_features, budget_select,
                                 calibrate_head, create_refiner,
                                 train_refiner, generate_weak_class_features,
                                 learn_proto_merge_threshold)
from client_stats import compute_client_stats, aggregate_client_stats

def sample_round_clients(args):
    select_clients = args.select_clients or args.num_clients
    select_clients = min(select_clients, args.num_clients)
    if select_clients == args.num_clients:
        return np.arange(args.num_clients)
    return np.random.choice(args.num_clients, size=select_clients, replace=False)


def local_train(head, features, labels, local_epochs, batch_size, lr, device):
    """Train head locally on client data. Returns trained head."""
    head = head.to(device)
    head.train()
    optimizer = optim.Adam(head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    dataset = FeatureDataset(features, labels, device=device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    best_state = None
    best_acc = -1.0

    for epoch in range(local_epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for feats, labs in loader:
            logits = head(feats)
            loss = criterion(logits, labs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * feats.size(0)
            correct += (logits.argmax(1) == labs).sum().item()
            total += labs.size(0)

        acc = correct / total if total > 0 else 0
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in head.state_dict().items()}

    if best_state is not None:
        head.load_state_dict(best_state)
    return head



def collect_static_client_stats(args, client_data, merge_threshold=None, return_raw=False):
    all_client_stats_list = []
    for k in range(args.num_clients):
        feats_k, labs_k = client_data[k]
        stats = compute_client_stats(
            feats_k, labs_k, args.num_classes,
            args.num_modes, args.device,
            min_samples=args.min_samples_per_class)
        all_client_stats_list.append(stats)

    threshold = args.proto_merge_threshold if merge_threshold is None else merge_threshold
    all_prototypes = aggregate_client_stats(
        all_client_stats_list, args.num_classes,
        args.feat_dim, args.num_modes, args.device,
        merge_sim_threshold=threshold,
        use_similarity_merge=getattr(args, "proto_similarity_merge", True))

    if return_raw:
        return all_prototypes, all_client_stats_list
    return all_prototypes


def _count_uploaded_prototypes(all_client_stats_list, num_classes):
    total = 0
    per_class = {}
    for c in range(num_classes):
        n_c = 0
        for stats in all_client_stats_list:
            n_c += len(stats["counts"].get(c, {}))
        per_class[c] = n_c
        total += n_c
    return total, per_class


def _count_merged_clusters(all_prototypes, num_classes):
    per_class = {c: len(all_prototypes.get(c, {})) for c in range(num_classes)}
    return sum(per_class.values()), per_class


def _log_proto_merge_summary(prefix, all_client_stats_list, all_prototypes, args, threshold):
    raw_total, raw_per_class = _count_uploaded_prototypes(all_client_stats_list, args.num_classes)
    merged_total, merged_per_class = _count_merged_clusters(all_prototypes, args.num_classes)
    mode = "similarity" if getattr(args, "proto_similarity_merge", True) else "off"
    print(f"  [ProtoMerge] {prefix}: mode={mode} threshold={threshold:.4f} "
          f"uploaded={raw_total} merged_clusters={merged_total} "
          f"reduction={raw_total - merged_total}")
    print(f"  [ProtoMerge] per_class uploaded={raw_per_class}")
    print(f"  [ProtoMerge] per_class merged={merged_per_class}")


def _step_adapt_proto_merge_threshold(args, threshold, refiner):
    metrics = getattr(refiner, "last_train_metrics", {}) if refiner is not None else {}
    proto_loss = metrics.get("proto")
    if proto_loss is None:
        return threshold, False

    new_threshold = threshold
    if proto_loss > args.proto_merge_target_high:
        new_threshold = min(args.proto_merge_threshold_max,
                            threshold + args.proto_merge_threshold_step)
    elif proto_loss < args.proto_merge_target_low:
        new_threshold = max(args.proto_merge_threshold_min,
                            threshold - args.proto_merge_threshold_step)

    changed = abs(new_threshold - threshold) > 1e-12
    direction = "unchanged"
    if changed:
        direction = "up" if new_threshold > threshold else "down"
    print(f"  [ProtoMerge] adaptive threshold {direction}: "
          f"proto_loss={proto_loss:.4f}, threshold={threshold:.4f}->{new_threshold:.4f}")
    return new_threshold, changed


def _adapt_proto_merge_threshold(
    args,
    threshold,
    refiner,
    all_client_stats_list,
    current_acc=None,
    best_acc=None,
    prev_update_acc=None,
):
    if not getattr(args, "proto_similarity_merge", True):
        return threshold, False
    if not getattr(args, "proto_merge_adaptive", True):
        return threshold, False
    if not getattr(args, "proto_merge_learnable", True):
        return _step_adapt_proto_merge_threshold(args, threshold, refiner)

    new_threshold, changed, info = learn_proto_merge_threshold(
        refiner, all_client_stats_list, args, threshold,
        current_acc=current_acc,
        best_acc=best_acc,
        prev_update_acc=prev_update_acc)

    direction = "unchanged"
    if changed:
        direction = "up" if new_threshold > threshold else "down"

    if info is None:
        print(f"  [ProtoMerge] learned threshold {direction}: "
              f"threshold={threshold:.4f}->{new_threshold:.4f}")
    else:
        best_text = "None" if info["best_acc"] is None else f"{info['best_acc']:.4f}"
        prev_text = "None" if info["prev_update_acc"] is None else f"{info['prev_update_acc']:.4f}"
        print(f"  [ProtoMerge] learned threshold {direction}: "
              f"proto_loss={info['proto_loss']:.4f} "
              f"acc={info['current_acc']:.4f} "
              f"best_acc={best_text} "
              f"prev_merge_acc={prev_text} "
              f"acc_delta={info['acc_delta']:.4f} "
              f"proto_q={info['proto_quality']:.3f} "
              f"acc_q={info['acc_quality']:.3f} "
              f"quality={info['quality']:.3f} "
              f"target={info['target_tau']:.4f} "
              f"loss={info['loss']:.4f} "
              f"pairs={info['pairs']} "
              f"threshold={threshold:.4f}->{new_threshold:.4f}")
    return new_threshold, changed

def evaluate_fedproref(model, features, labels, device, args):
    metrics = evaluate_all(model, features, labels, device)
    domain_ids = getattr(args, "test_domain_ids", None)
    domain_names = getattr(args, "test_domain_names", None)
    if domain_ids is None or not domain_names:
        return metrics

    if not torch.is_tensor(domain_ids):
        domain_ids = torch.as_tensor(domain_ids, dtype=torch.long)
    domain_ids = domain_ids.to(labels.device)
    if domain_ids.numel() != labels.numel():
        print("  [Eval] Ignoring domain-average metrics: test_domain_ids size mismatch")
        return metrics

    domain_accs = []
    for domain_idx, domain_name in enumerate(domain_names):
        mask = domain_ids == domain_idx
        if mask.sum().item() == 0:
            continue
        domain_metrics = evaluate_all(model, features[mask], labels[mask], device)
        key = str(domain_name).lower().replace("-", "_").replace(" ", "_")
        metrics[f"acc_{key}"] = domain_metrics["acc"]
        domain_accs.append(domain_metrics["acc"])

    if domain_accs:
        metrics["acc_sample"] = metrics["acc"]
        metrics["acc_domain_avg"] = float(np.mean(domain_accs))
        metrics["acc"] = metrics["acc_domain_avg"]
    return metrics


def run_fedavg(args, client_data, test_features, test_labels, logger):
    """Baseline: FedAvg with frozen backbone, head-only training."""
    device = args.device
    global_head = create_head(args.head_type, args.feat_dim,
                              args.num_classes, args.head_hidden).to(device)

    for rnd in range(1, args.comm_rounds + 1):
        round_start = time.perf_counter()
        client_heads = []
        client_weights = []
        selected_clients = sample_round_clients(args)

        for k in selected_clients:
            feats_k, labs_k = client_data[k]
            local_head = clone_head(global_head)
            local_head = local_train(
                local_head, feats_k, labs_k,
                args.local_epochs, args.batch_size, args.lr, device)
            client_heads.append(local_head)
            client_weights.append(len(labs_k))

        # Weighted FedAvg
        total = sum(client_weights)
        weights = [w / total for w in client_weights]
        global_head = fedavg_heads(global_head, client_heads, weights)

        # Evaluate
        metrics = evaluate_fedproref(global_head, test_features, test_labels, device, args)
        metrics["round_time"] = time.perf_counter() - round_start
        logger.log(rnd, metrics)

    return global_head


def run_fedproref(args, client_data, test_features, test_labels, logger):
    """
    FedProRef pipeline (also handles ablation methods: proto_aug, proto_cal, proto_sample):

    fedproref (main method):
      1. One-time client upload of original-feature prototypes and counts
      2. Each round: weak-class augmentation (via refiner) + local train → upload heads
      3. Server: FedAvg heads, finetune refiner on cached statistics, then broadcast

    Ablation methods:
      - proto_aug:    Gaussian augmentation only (no refiner)
      - proto_cal:    Prototype-based head calibration (no sampling, no refiner)
      - proto_sample: Prototype mixture sampling + head calibration (no refiner)
    """
    device = args.device
    global_head = create_head(args.head_type, args.feat_dim,
                              args.num_classes, args.head_hidden).to(device)

    refiner = None
    prev_all_prototypes = None
    all_prototypes = None
    cached_aug_features = {}      # {client_id: (aug_feats, aug_labs)}
    refiner_updated_last_round = False

    proto_merge_threshold = args.proto_merge_threshold
    best_acc_so_far = None
    last_proto_merge_acc = None

    print("\nCollecting one-time client statistics from frozen original features...")
    all_prototypes, all_client_stats_list = collect_static_client_stats(
        args, client_data, proto_merge_threshold, return_raw=True)
    _log_proto_merge_summary("initial", all_client_stats_list, all_prototypes,
                             args, proto_merge_threshold)
    prev_all_prototypes = all_prototypes
    print("  Static client statistics cached. Later rounds upload heads only.\n")

    # ══════════════════════════════════════════════════════════════════════
    # Phase 0: Pretrain Refiner (fedproref only)
    # ══════════════════════════════════════════════════════════════════════
    if args.method == "fedproref" and args.refiner_type != "none" and args.refiner_pretrain_epochs > 0:
        print(f"\n{'='*70}")
        print(f"  Phase 0: Pretraining Refiner ({args.refiner_pretrain_epochs} epochs)")
        print(f"{'='*70}")

        # Create and pretrain refiner
        refiner = create_refiner(args)
        print(f"  Pretraining Refiner with {args.refiner_pretrain_epochs} epochs...")
        refiner = train_refiner(
            refiner=refiner,
            all_prototypes=all_prototypes,
            num_classes=args.num_classes,
            gen_per_class=args.gen_per_class,
            proposal_sigma=args.proposal_sigma,
            refiner_epochs=args.refiner_pretrain_epochs,
            refiner_lr=args.refiner_lr,
            w_reg=args.w_reg,
            w_proto=args.w_proto,
            w_flow=args.w_flow,
            device=device,
            refiner_type=args.refiner_type,
        )
        print("  [ProtoMerge] threshold update skipped after pretrain: "
              "waiting for global acc feedback")
        prev_all_prototypes = all_prototypes
        print(f"  Refiner pretraining completed.\n")

    # ══════════════════════════════════════════════════════════════════════
    # Federated Learning Rounds
    # ══════════════════════════════════════════════════════════════════════
    for rnd in range(1, args.comm_rounds + 1):
        round_start = time.perf_counter()
        client_heads = []
        client_weights = []
        selected_clients = sample_round_clients(args)
        threshold_feedback_ready = False

        # ── Phase 1: Client local training + head upload ──
        for k in selected_clients:
            feats_k, labs_k = client_data[k]

            # Augment weak classes (method-dependent)
            train_feats, train_labs = feats_k, labs_k

            if args.method == "proto_aug" and prev_all_prototypes is not None:
                # Ablation: Gaussian augmentation only (no refiner)
                if k not in cached_aug_features:
                    aug_feats, aug_labs = generate_weak_class_features(
                        None, labs_k, prev_all_prototypes,
                        args.min_samples_per_class, args.num_classes,
                        args.aug_gen_per_class,
                        args.proposal_sigma, device,
                        weak_percentile=args.weak_class_percentile,
                        use_refiner=False)
                    cached_aug_features[k] = (aug_feats, aug_labs)
                else:
                    aug_feats, aug_labs = cached_aug_features[k]

                if len(aug_feats) > 0:
                    train_feats = np.concatenate([feats_k, aug_feats.numpy()], axis=0)
                    train_labs  = np.concatenate([labs_k,  aug_labs.numpy()],  axis=0)

            elif args.method == "fedproref" and refiner is not None and prev_all_prototypes is not None:
                # FedProRef: refiner-based augmentation
                if refiner_updated_last_round or k not in cached_aug_features:
                    aug_feats, aug_labs = generate_weak_class_features(
                        refiner, labs_k, prev_all_prototypes,
                        args.min_samples_per_class, args.num_classes,
                        args.aug_gen_per_class,
                        args.proposal_sigma, device,
                        weak_percentile=args.weak_class_percentile)
                    cached_aug_features[k] = (aug_feats, aug_labs)
                else:
                    aug_feats, aug_labs = cached_aug_features[k]

                if len(aug_feats) > 0:
                    train_feats = np.concatenate([feats_k, aug_feats.numpy()], axis=0)
                    train_labs  = np.concatenate([labs_k,  aug_labs.numpy()],  axis=0)

            # Local training (standard FedAvg)
            local_head = clone_head(global_head)
            local_head = local_train(
                local_head, train_feats, train_labs,
                args.local_epochs, args.batch_size, args.lr, device)
            client_heads.append(local_head)
            client_weights.append(len(labs_k))

        # ── Phase 2: FedAvg aggregation ──
        total = sum(client_weights)
        weights = [w / total for w in client_weights]
        global_head = fedavg_heads(global_head, client_heads, weights)

        # ── Phase 3: Server-side (method-dependent) ──
        if all_prototypes is not None:
            if args.method == "proto_aug":
                # Static prototypes are cached; augmentation cache stays valid.
                pass

            elif args.method == "proto_cal":
                # Ablation: calibrate head directly with prototypes
                proto_feats = []
                proto_labels = []
                for c in range(args.num_classes):
                    if c not in all_prototypes:
                        continue
                    for m in all_prototypes[c]:
                        if len(all_prototypes[c][m]) == 0:
                            continue
                        for mu, n in all_prototypes[c][m]:
                            proto_feats.append(mu.unsqueeze(0))
                            proto_labels.append(torch.tensor([c]))
                if len(proto_feats) > 0:
                    proto_feats = torch.cat(proto_feats)
                    proto_labels = torch.cat(proto_labels)
                    cal_head = calibrate_head(
                        global_head, proto_feats, proto_labels,
                        args.cal_epochs, args.cal_lr, device)
                    global_head = fuse_head(global_head, cal_head, args.beta_head)

            elif args.method == "proto_sample":
                # Ablation: Gaussian sampling + head calibration (no refiner)
                gen_feats, gen_labels = generate_calibration_features(
                    None, all_prototypes, args.num_classes,
                    args.gen_per_class, args.proposal_sigma, device,
                    use_refiner=False)
                sel_feats, sel_labels = budget_select(
                    gen_feats, gen_labels, args.cal_budget, args.num_classes)
                cal_head = calibrate_head(
                    global_head, sel_feats, sel_labels,
                    args.cal_epochs, args.cal_lr, device)
                global_head = fuse_head(global_head, cal_head, args.beta_head)

            elif args.method == "fedproref":
                # FedProRef ablation: refiner_type=none keeps prototype upload but skips refiner updates.
                if args.refiner_type != "none":
                    if refiner is None:
                        refiner = create_refiner(args)

                    do_finetune = (args.cal_every > 0) and (rnd % args.cal_every == 0)
                    if do_finetune:
                        print(f"  [Round {rnd}] Server: Finetune Refiner ({args.refiner_finetune_epochs} epochs) → broadcast")
                        refiner = train_refiner(
                            refiner=refiner,
                            all_prototypes=all_prototypes,
                            num_classes=args.num_classes,
                            gen_per_class=args.gen_per_class,
                            proposal_sigma=args.proposal_sigma,
                            refiner_epochs=args.refiner_finetune_epochs,
                            refiner_lr=args.refiner_lr,
                            w_reg=args.w_reg,
                            w_proto=args.w_proto,
                            w_flow=args.w_flow,
                            device=device,
                            refiner_type=args.refiner_type,
                        )
                        threshold_feedback_ready = True
                        prev_all_prototypes = all_prototypes
                        refiner_updated_last_round = True
                        cached_aug_features.clear()
                    else:
                        refiner_updated_last_round = False
                else:
                    refiner_updated_last_round = False

        # ── Phase 4: Evaluate ──
        metrics = evaluate_fedproref(global_head, test_features, test_labels, device, args)
        metrics["round_time"] = time.perf_counter() - round_start
        logger.log(rnd, metrics)

        current_acc = metrics.get("acc")
        if args.method == "fedproref" and threshold_feedback_ready and current_acc is not None:
            proto_merge_threshold, merge_changed = _adapt_proto_merge_threshold(
                args, proto_merge_threshold, refiner, all_client_stats_list,
                current_acc=current_acc,
                best_acc=best_acc_so_far,
                prev_update_acc=last_proto_merge_acc)
            last_proto_merge_acc = float(current_acc)
            if merge_changed:
                all_prototypes = aggregate_client_stats(
                    all_client_stats_list, args.num_classes,
                    args.feat_dim, args.num_modes, args.device,
                    merge_sim_threshold=proto_merge_threshold,
                    use_similarity_merge=getattr(args, "proto_similarity_merge", True))
                _log_proto_merge_summary(f"after round {rnd}",
                                         all_client_stats_list,
                                         all_prototypes, args,
                                         proto_merge_threshold)
                prev_all_prototypes = all_prototypes
                cached_aug_features.clear()

        if current_acc is not None:
            current_acc = float(current_acc)
            best_acc_so_far = current_acc if best_acc_so_far is None else max(best_acc_so_far, current_acc)

    return global_head


def main():
    args = get_args()
    set_seed(args.seed)

    # Logger — 创建后立即 tee stdout，之后所有 print 都同步写入文件
    logger = Logger(args.log_dir, f"{args.exp_name}_{args.method}_{args.dataset}_a{args.alpha}")
    logger.log_params(args)

    print("=" * 70)
    print(f"  FedProRef Experiment")
    print(f"  Method:   {args.method}")
    print(f"  Dataset:  {args.dataset} (alpha={args.alpha})")
    print(f"  Clients:  {args.num_clients}")
    print(f"  Selected: {args.select_clients or args.num_clients}")
    print(f"  Rounds:   {args.comm_rounds}")
    print(f"  Refiner:  {args.refiner_type}")
    print(f"  Device:   {args.device}")
    print("=" * 70)

    # Load data
    client_data, test_features, test_labels = get_or_extract_features(args)
    print(f"\nTest set: {test_features.shape[0]} samples, {args.num_classes} classes")

    # Pre-move test data to device once
    device = args.device
    test_features = test_features.to(device)
    test_labels = test_labels.to(device)

    # Run
    if args.method == "fedavg":
        model = run_fedavg(args, client_data, test_features, test_labels, logger)
    else:
        model = run_fedproref(args, client_data, test_features, test_labels, logger)

    # Report best
    best = logger.best("acc")
    print(f"\n{'=' * 70}")
    print(f"  Best: Round {best['round']} | Acc={best['acc']:.2f}% | ECE={best['ece']:.4f} | NLL={best['nll']:.4f}")
    print(f"{'=' * 70}")

    # Save model
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"{args.exp_name}_{args.method}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    logger.close()
    return {
        "best_round": best["round"],
        "best_acc": best["acc"],
        "best_ece": best["ece"],
        "best_nll": best["nll"],
    }


if __name__ == "__main__":
    main()
