# FedProRef

**Server-Trained Prototype Feature Refinement for Federated Learning under Severe Label Skew**

FedProRef is a feature-space federated learning framework for severe non-IID label skew. It freezes a pretrained OpenCLIP visual encoder, federates only a lightweight classifier head, and uses compact client-uploaded prototype/count statistics to train a server-side class-conditioned MLP refiner. The trained refiner is then used by clients to synthesize feature-label pairs for locally weak or missing classes before local head training.

The implementation is a single-machine simulation of the federated protocol. Raw images, raw feature sets, covariance matrices, logs, checkpoints, and feature caches are not intended to be committed to GitHub.

## Method Summary

FedProRef follows this pipeline:

1. Extract frozen OpenCLIP image features and cache them locally.
2. Split the training set across clients using Dirichlet label skew.
3. Each client uploads only eligible class prototypes and counts.
4. The server builds a class-wise prototype pool and trains a class-conditioned MLP refiner.
5. In each communication round, clients train local classifier heads on real cached features plus refiner-generated weak-class synthetic features.
6. The server aggregates classifier heads with FedAvg and periodically finetunes the refiner from the cached prototype pool.

The final method in the paper is `FedProRef` with `--refiner_type mlp`. The RF-style refiner is kept as an auxiliary ablation via `--refiner_type rf`; it is not the final reported method.

## Repository Layout

```text
.
├── config.py                  # command-line configuration
├── data_utils.py              # dataset loading, Dirichlet split, OpenCLIP feature cache
├── client_stats.py            # client prototype/count statistics
├── server_calibration.py      # prototype pool, refiner training, feature synthesis
├── refiner_mlp.py             # class-conditioned residual MLP refiner
├── refiner_rf.py              # RF-style auxiliary refiner ablation
├── proposal.py                # prototype proposal sampling
├── head.py                    # linear/MLP classifier head and FedAvg utilities
├── federated_loop.py          # main federated training loop
├── run.py                     # repeated-run wrapper
├── summarize_results.py       # log/result summarization utilities
├── visualize.py               # plotting utilities
├── train_plan_*.sh            # experiment-plan scripts
└── paper/                     # manuscript source and figures
```

## Environment

Python 3.9+ and PyTorch are recommended. A minimal environment is:

```bash
pip install torch torchvision
pip install open-clip-torch scikit-learn numpy pandas matplotlib openpyxl
```

For GPU runs, install the PyTorch build matching your CUDA version before installing the remaining packages.

## Data and Checkpoints

The code supports CIFAR-10, CIFAR-100, and Tiny-ImageNet. CIFAR-10/100 can be downloaded automatically by `torchvision`; Tiny-ImageNet should be placed under `data/tiny-imagenet-200/` or provided as `data/tiny-imagenet-200.zip` depending on your local setup.

Large runtime artifacts are intentionally excluded from GitHub:

```text
data/
logs/
checkpoints/
results/
pretrain_path/
*.npz
*.pth
*.safetensors
```

The paper experiments used OpenCLIP ViT-B/16 with feature dimension `d=512`. You can either use an OpenCLIP pretrained tag such as `openai`, or place a local checkpoint under `pretrain_path/` and pass it with `--pretrained`.

## Quick Start

Run a small FedAvg baseline:

```bash
python federated_loop.py \
  --method fedavg \
  --dataset cifar10 \
  --alpha 0.1 \
  --num_clients 10 \
  --select_clients 10 \
  --comm_rounds 5 \
  --local_epochs 1 \
  --backbone ViT-B-16 \
  --pretrained openai \
  --device auto \
  --exp_name quick_fedavg
```

Run the main FedProRef method:

```bash
python federated_loop.py \
  --method fedproref \
  --refiner_type mlp \
  --dataset cifar100 \
  --alpha 0.01 \
  --num_clients 10 \
  --select_clients 10 \
  --comm_rounds 100 \
  --local_epochs 10 \
  --backbone ViT-B-16 \
  --pretrained openai \
  --feat_dim 512 \
  --num_modes 1 \
  --proposal_sigma 0.05 \
  --gen_per_class 500 \
  --aug_gen_per_class 100 \
  --weak_class_percentile 10 \
  --min_samples_per_class 10 \
  --refiner_pretrain_epochs 300 \
  --refiner_finetune_epochs 50 \
  --cal_every 10 \
  --w_proto 0.1 \
  --w_reg 0.01 \
  --seed 42 \
  --partition_seed 42 \
  --device auto \
  --exp_name fedproref_cifar100_a001_s42
```

Run the repeated-run wrapper:

```bash
python run.py \
  --method fedproref \
  --refiner_type mlp \
  --dataset cifar100 \
  --alpha 0.01 \
  --repeats 3 \
  --seed 42 \
  --partition_seed 42 \
  --pretrained openai
```

## Methods and Ablations

| Command | Description |
| --- | --- |
| `--method fedavg` | Head-only FedAvg baseline with frozen OpenCLIP features. |
| `--method proto_cal` | Server-side prototype-mean head calibration. |
| `--method proto_aug` | Client-side weak-class augmentation with naive Gaussian prototype proposals. |
| `--method proto_sample` | Server-side calibration using Gaussian-sampled prototype features. |
| `--method fedproref --refiner_type mlp` | Main FedProRef method. |
| `--method fedproref --refiner_type rf` | Auxiliary RF-style refiner ablation. |
| `--method fedproref --refiner_type mlp --no_proto_similarity_merge` | FedProRef without same-class prototype similarity merging. |

## Main Experimental Settings

The main paper protocol uses:

| Item | Setting |
| --- | --- |
| Feature extractor | OpenCLIP ViT-B/16 |
| Feature dimension | 512 |
| Feature handling | cached frozen features; L2 normalized after `encode_image()` |
| Classifier head | `Linear(512, C)` by default |
| Clients | 10 total, 10 selected per round |
| Participation | full participation |
| Communication rounds | 100 |
| Local epochs | 10 |
| Batch size | 64 |
| Local optimizer | Adam, learning rate `1e-3`, weight decay `0` |
| Non-IID split | Dirichlet label skew on the training split |
| Partition seed | 42 |
| Training seeds | 42, 43, 44 |
| Main alpha values | 0.01, 0.03, 0.05 |
| Extended alpha values | 0.01, 0.03, 0.05, 0.07, 0.09, 0.10, 0.30, 0.50 |

FedProRef-specific defaults used by the main experiments:

| Item | Setting |
| --- | --- |
| Refiner | class-conditioned residual MLP |
| Class embedding dim | 512 |
| Refiner input | proposal feature plus class embedding, dimension 1024 |
| Hidden dim / layers | 512 / 3 linear layers |
| Activation | ReLU |
| Refined feature | `Normalize(z0 + r_phi(z0, e_c))` |
| Refiner optimizer | Adam, learning rate `1e-3` |
| Pretrain / finetune | 300 epochs / 50 epochs |
| Refiner update period | every 10 rounds |
| Proposal noise | `sigma = 0.05` |
| Server proposals per class | `gen_per_class = 500` |
| Synthetic features per weak class | `aug_gen_per_class = 100` |
| Weak-class threshold | class count `< 10`, plus bottom 10 percentile |
| Prototype merge threshold | initial 0.90, adaptive range [0.70, 0.98] |
| Loss weights | `lambda_proto = 0.1`, `lambda_reg = 0.01` |

## Reproducing Experiment Plans

The shell scripts define larger experiment sweeps. For example:

```bash
bash train_plan_cifar_tiny.sh
```

Useful environment overrides:

```bash
DRY_RUN=1 bash train_plan_cifar_tiny.sh
MAX_JOBS=3 bash train_plan_cifar_tiny.sh
METHODS="fedproref proto_aug proto_cal" bash train_plan_cifar_tiny.sh
FEDPROREF_REFINER_TYPES="mlp" bash train_plan_cifar_tiny.sh
```

## Outputs

Runtime outputs are written to:

```text
logs/          # training logs
checkpoints/   # saved classifier heads
results/       # parsed summaries and spreadsheets
data/*_cache/  # extracted OpenCLIP feature caches
```

These directories are ignored by `.gitignore` and should not be uploaded unless you intentionally publish a separate release artifact.

## Citation

If you use this code, cite the corresponding FedProRef manuscript:

```bibtex
@article{wang2026fedproref,
  title  = {FedProRef: Server-Trained Prototype Feature Refinement for Federated Learning under Severe Label Skew},
  author = {Wang, Lei},
  year   = {2026}
}
```
