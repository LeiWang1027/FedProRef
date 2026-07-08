"""
FedProRef: Configuration.
All hyperparameters in one place for easy ablation.
"""
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="FedProRef")

    # ── Dataset & Partition ──────────────────────────────────────────
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "cifar100", "tinyimagenet", "pacs", "officehome"])
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--select_clients", type=int, default=None,
                        help="Selected clients per communication round; default uses all clients")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Dirichlet concentration (smaller = more heterogeneous)")
    parser.add_argument("--min_require_size", type=int, default=2)
    parser.add_argument("--min_samples_per_class", type=int, default=10,
                        help="Skip class stats upload if client has fewer than N samples of that class")

    # ── Backbone ─────────────────────────────────────────────────────
    parser.add_argument("--backbone", type=str, default="ViT-B-32",
                        help="OpenCLIP model name")
    parser.add_argument("--pretrained", type=str, default="openai",
                        help="Pretrained weights tag or local path")
    parser.add_argument("--feat_dim", type=int, default=512)

    # ── Head ─────────────────────────────────────────────────────────
    parser.add_argument("--head_type", type=str, default="linear",
                        choices=["linear", "mlp"])
    parser.add_argument("--head_hidden", type=int, default=512)

    # ── Federated Learning ───────────────────────────────────────────
    parser.add_argument("--comm_rounds", type=int, default=100)
    parser.add_argument("--local_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta_head", type=float, default=0.2,
                        help="Head fusion coefficient: W = (1-beta)*W_local + beta*W_cal")

    # ── Multi-mode decomposition ─────────────────────────────────────
    parser.add_argument("--num_modes", type=int, default=2,
                        help="Number of k-means clusters per class per client")
    parser.add_argument("--proto_merge_threshold", type=float, default=0.90,
                        help="Initial cosine threshold for prototype-cluster merging")
    parser.add_argument("--no_proto_similarity_merge", dest="proto_similarity_merge",
                        action="store_false",
                        help="Disable cosine-similarity prototype merging; keep uploaded prototypes separate")
    parser.add_argument("--no_proto_merge_adaptive", dest="proto_merge_adaptive",
                        action="store_false",
                        help="Disable refiner-feedback adaptation of prototype merge threshold")
    parser.set_defaults(proto_merge_adaptive=True, proto_similarity_merge=True)
    parser.add_argument("--proto_merge_threshold_min", type=float, default=0.70)
    parser.add_argument("--proto_merge_threshold_max", type=float, default=0.98)
    parser.add_argument("--proto_merge_threshold_step", type=float, default=0.03)
    parser.add_argument("--proto_merge_target_low", type=float, default=0.04)
    parser.add_argument("--proto_merge_target_high", type=float, default=0.10)
    parser.add_argument("--no_proto_merge_learnable", dest="proto_merge_learnable",
                        action="store_false",
                        help="Disable gradient-learned prototype merge threshold")
    parser.set_defaults(proto_merge_learnable=True)
    parser.add_argument("--proto_merge_threshold_lr", type=float, default=0.05,
                        help="Learning rate for the prototype merge threshold parameter")
    parser.add_argument("--proto_merge_temperature", type=float, default=0.05,
                        help="Temperature for soft prototype merge gates used to learn the threshold")
    parser.add_argument("--proto_merge_learn_steps", type=int, default=10,
                        help="Gradient steps for learning the prototype merge threshold after each refiner update")
    parser.add_argument("--proto_merge_tau_reg", type=float, default=0.1,
                        help="Regularization weight pulling learned threshold toward the refiner-quality target")
    parser.add_argument("--proto_merge_acc_weight", type=float, default=0.5,
                        help="Weight of global accuracy feedback when learning the prototype merge threshold")
    parser.add_argument("--proto_merge_acc_drop_tolerance", type=float, default=0.3,
                        help="Accuracy drop from historical best, in percentage points, that blocks looser merging")
    parser.add_argument("--proto_merge_acc_gain_tolerance", type=float, default=0.2,
                        help="Accuracy change from previous threshold update, in percentage points, treated as strong feedback")

    # ── Proposal ─────────────────────────────────────────────────────
    parser.add_argument("--proposal_sigma", type=float, default=0.05,
                        help="Noise scale for prototype mixture proposal")

    # ── Refiner ──────────────────────────────────────────────────────
    parser.add_argument("--refiner_type", type=str, default="rf",
                        choices=["none", "mlp", "rf"],
                        help="none=no refiner, mlp=MLP-refiner, rf=Rectified Flow")
    parser.add_argument("--refiner_hidden", type=int, default=512)
    parser.add_argument("--refiner_layers", type=int, default=3)
    parser.add_argument("--rf_steps", type=int, default=4,
                        help="Euler integration steps for RF")
    parser.add_argument("--refiner_pretrain_epochs", type=int, default=200,
                        help="Refiner pretraining epochs (before federated rounds)")
    parser.add_argument("--refiner_finetune_epochs", type=int, default=50,
                        help="Refiner finetuning epochs (during federated rounds)")
    parser.add_argument("--refiner_lr", type=float, default=1e-3)
    parser.add_argument("--cal_every", type=int, default=10,
                        help="Finetune refiner every N communication rounds (0=no finetuning)")

    # ── Loss weights ─────────────────────────────────────────────────
    parser.add_argument("--w_reg", type=float, default=0.01)
    parser.add_argument("--w_proto", type=float, default=0.1)
    parser.add_argument("--w_flow", type=float, default=1.0,
                        help="Flow matching loss weight for RF refiner")

    # ── Budgeted calibration ─────────────────────────────────────────
    parser.add_argument("--gen_per_class", type=int, default=500,
                        help="Candidate features generated per class (for server calibration)")
    parser.add_argument("--aug_gen_per_class", type=int, default=100,
                        help="Synthetic samples per weak class (for local training augmentation)")
    parser.add_argument("--weak_class_percentile", type=float, default=0.0,
                        help="Bottom X%% of classes (by sample count) are also treated as weak. "
                             "0=disabled. E.g. 20 means the 20%% least-represented classes trigger synthesis.")


    parser.add_argument("--cal_budget", type=int, default=500,
                        help="Total calibration budget B")
    parser.add_argument("--cal_epochs", type=int, default=20)
    parser.add_argument("--cal_lr", type=float, default=1e-3)

    # ── Method ablation switches ─────────────────────────────────────
    parser.add_argument("--method", type=str, default="fedproref",
                        choices=["fedavg", "proto_aug", "proto_cal", "proto_sample", "fedproref"],
                        help="Which method to run (for ablation)")

    # ── Misc ─────────────────────────────────────────────────────────
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--partition_seed", type=int, default=42,
                        help="Seed for data partition/cache; keep fixed when comparing training seeds")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--exp_name", type=str, default="default")

    args = parser.parse_args()


    if args.device == "auto":
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Derive num_classes
    if args.dataset == "cifar10":
        args.num_classes = 10
    elif args.dataset == "cifar100":
        args.num_classes = 100
    elif args.dataset == "tinyimagenet":
        args.num_classes = 200
    elif args.dataset == "pacs":
        args.num_classes = 7
        if args.num_clients != 4:
            print("[Config] PACS uses one client per domain; overriding num_clients to 4")
            args.num_clients = 4
    elif args.dataset == "officehome":
        args.num_classes = 65
        if args.num_clients != 4:
            print("[Config] OfficeHome uses one client per domain; overriding num_clients to 4")
            args.num_clients = 4

    return args
