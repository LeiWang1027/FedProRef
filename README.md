# FedProRef

**Communication-Efficient Prototype-Count Supervision for Federated Learning Under Severe Label Skew**

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

## Evidence Scope

The reported implementation freezes the representation and federates only the
classifier head. The evidence covers OpenCLIP ViT-B/16 and RN50; it does not
claim transfer to jointly updated or heterogeneous client encoders. Both
client-class floor settings are exposed: `min_require_size=0` exercises truly
missing local classes while repairing only globally empty clients, and
`min_require_size=1` retains the protocol-aligned floor controls. The
DirectAnchorAug control isolates the benefit of clean shared anchors; the
refiner-specific claim is the correction of noisy prototype proposals relative
to ProtoAug, not universal superiority over clean-anchor augmentation.

## Repository Layout

```text
.
├── config.py                  # command-line configuration
├── data_utils.py              # dataset loading, Dirichlet split, OpenCLIP feature cache
├── backbone_utils.py          # checkpoint identity and dynamic feature dimensions
├── client_stats.py            # client prototype/count statistics
├── server_calibration.py      # prototype pool, refiner training, feature synthesis
├── direct_anchor_aug.py       # clean-anchor matched control
├── refiner_mlp.py             # class-conditioned residual MLP refiner
├── refiner_rf.py              # RF-style auxiliary refiner ablation
├── proposal.py                # prototype proposal sampling
├── head.py                    # linear/MLP classifier head and FedAvg utilities
├── federated_loop.py          # main federated training loop
├── run.py                     # repeated-run wrapper
├── run_missing_class_24.sh    # focused missing-class stress test
├── run_fedavg_vit_alpha_144.sh # FedAvg floor/alpha control matrix
├── run_vit_minpc0_methods_432.sh # active no-floor method matrix (171 runs)
├── run_backbone_rn50_48.sh    # fixed 48-run RN50 robustness matrix
├── summarize_*.py             # detailed, aggregate, and paired summaries
├── partition_report.py        # client/class partition audit report
├── scripts/download_checkpoints.py # verified checkpoint downloader
├── summarize_results.py       # log/result summarization utilities
├── visualize.py               # plotting utilities
└── tests/                     # functional and experiment-contract tests
```

## Environment

The reference experiments used Python 3.11 and OpenCLIP 3.3.0. Install a
PyTorch build compatible with the local CUDA driver, then install the remaining
dependencies:

```bash
pip install torch torchvision
pip install -r requirements.txt
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

The primary checkpoint is the public OpenCLIP ViT-B/16 OpenAI checkpoint
(`timm/vit_base_patch16_clip_224.openai/open_clip_model.safetensors`), with
SHA-256 `4b8699299b1e8997753c64b052ba32031449d5d853f55a039148560ee02b820f`.
The RN50 study uses the official OpenAI checkpoint with SHA-256
`afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762`.
Download and verify both without committing the large files:

```bash
python scripts/download_checkpoints.py --checkpoint all
```

The files are written under the ignored `pretrain_path/` directory. You may
also pass an OpenCLIP registered tag such as `openai` to a one-off run, but the
fixed paper scripts deliberately verify the exact local checkpoint hashes.
Cache identities include the backbone, checkpoint hash, dataset, alpha,
partition seed, and `min_require_size`, so incompatible features are not
silently reused.

## Quick Start

Run a small functional FedAvg baseline (OpenCLIP downloads the registered
checkpoint on first use):

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

Run the main FedProRef method with the exact paper checkpoint:

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
  --pretrained ./pretrain_path/old_open_clip_model.safetensors \
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
| `--method direct_anchor_aug` | Client weak-class augmentation using count-weighted clean anchors only; no proposal noise and no refiner. |
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
| Client-class floor | both `min_require_size=0` and `1`; CLI default is `0` |
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

All fixed experiment runners support a non-training `plan` or `list` action
and a `command RUN_ID` action, so the complete matrix can be inspected before
launching it. Set `FEDPROREF_DATA_ROOT`, `FEDPROREF_PYTHON_BIN`, or the
script-specific results variable to use external data, Python, and result
locations. Training artifacts remain outside version control.

### Focused missing-class stress test

The revision-specific 24-run matrix uses `min_require_size=0`, guarantees that
every client is nonempty, and compares ProtoAug with FedProRef:

```bash
bash run_missing_class_24.sh list all
bash run_missing_class_24.sh command mcstress_cifar100_a001_ps42_ts42_fedproref
bash run_missing_class_24.sh all
```

The matrix contains 18 CIFAR-100 runs (three partition seeds, three training
seeds, two methods) and six Tiny-ImageNet runs (partition seed 42, three
training seeds, two methods), all at `alpha=0.01`.

### FedAvg ViT alpha/client-class sweep

The fixed FedAvg baseline matrix uses the verified local OpenCLIP ViT-B/16
checkpoint at `pretrain_path/old_open_clip_model.safetensors` and contains:

```text
3 datasets × 8 alpha values × 2 min_require_size values × 3 training seeds = 144 runs
```

Datasets are CIFAR-10, CIFAR-100, and Tiny-ImageNet. Alpha values are `0.01`,
`0.03`, `0.05`, `0.07`, `0.09`, `0.10`, `0.30`, and `0.50`; training seeds are
42, 43, and 44; `partition_seed=42` is fixed. `min_require_size=0` permits
missing client-class cells, while `min_require_size=1` guarantees at least one
sample in every client-class cell.

Inspect the matrix or one rendered command without starting training:

```bash
bash run_fedavg_vit_alpha_144.sh plan
bash run_fedavg_vit_alpha_144.sh command \
  vit_fedavg_cifar100_a010_minpc1_ps42_ts43
```

Launch all 144 jobs sequentially with completion markers and automatic resume:

```bash
bash run_fedavg_vit_alpha_144.sh all
```

The `all` action is the only command above that launches training. Results are
isolated under `results/fedavg_vit_alpha_144/`; rerunning `all` skips jobs with
a valid `.done` marker and completed Round 100 log record.

### RN50 backbone robustness experiment

The fixed robustness matrix changes only the frozen backbone to OpenCLIP RN50/OpenAI and compares both partition constraints:

```text
2 datasets × 2 min_require_size values × 4 methods × 3 training seeds = 48 runs
```

The methods are `fedavg`, `proto_aug`, `direct_anchor_aug`, and `fedproref`; datasets are CIFAR-100 and Tiny-ImageNet; `partition_seed=42`; training seeds are 42, 43, and 44. `min_require_size=0` allows missing client-class cells while keeping every client nonempty, whereas `min_require_size=1` guarantees at least one sample in every client-class cell. Both conditions use separate feature caches.

Inspect the exact matrix and any rendered command without starting training:

```bash
bash run_backbone_rn50_48.sh plan
bash run_backbone_rn50_48.sh command rn50_cifar100_a001_minpc1_ps42_ts42_fedproref
```

Run all 48 jobs sequentially with completion markers and automatic resume:

```bash
bash run_backbone_rn50_48.sh all
```

By default, data are read from `data/` and results are isolated under
`results/backbone_rn50_48/`. Override them when needed:

```bash
FEDPROREF_DATA_ROOT=/path/to/data \
FEDPROREF_RN50_RESULTS_ROOT=/path/to/results \
FEDPROREF_RN50_PRETRAINED=/path/to/RN50_openai.pt \
bash run_backbone_rn50_48.sh all
```

RN50 uses the official OpenCLIP preprocessing, L2-normalized features, and automatic feature-dimension resolution. Do not pass `--feat_dim` unless intentionally using it as a consistency assertion. A direct one-off run through the regular wrapper is also supported:

```bash
python run.py --dataset cifar100 --alpha 0.01 --method fedproref \
  --backbone RN50 --pretrained ./pretrain_path/RN50_openai.pt --min_require_size 1 \
  --partition_seed 42 --seed 42 --repeats 1
```

Regenerate the detailed, aggregate, and paired-difference reports after any completed runs:

```bash
bash run_backbone_rn50_48.sh summarize
```

This writes `runs.csv`, `aggregate.csv`, `paired_deltas.csv`, and `summary.md`. Aggregates use the sample standard deviation over seeds 42/43/44; paired differences always match dataset, partition floor, partition seed, and training seed.

### Full no-floor ViT-B/16 method matrix

The active no-floor runner covers FedProRef, ProtoAug, and DirectAnchorAug for
CIFAR-10/CIFAR-100 at eight alpha values and Tiny-ImageNet at three alpha
values, with three training seeds and partition seed 42 (171 runs total):

```bash
bash run_vit_minpc0_methods_432.sh plan
MAX_JOBS=3 bash run_vit_minpc0_methods_432.sh all
python summarize_vit_minpc0_methods_171.py
```

Set `MAX_JOBS=1` for serial execution. The script retains its historical name
for traceability; its emitted and validated active matrix contains 171 runs.

## Outputs

Runtime outputs are written to:

```text
logs/          # training logs
checkpoints/   # saved classifier heads
results/       # parsed summaries and spreadsheets
data/*_cache/  # extracted OpenCLIP feature caches
```

These directories are ignored by `.gitignore` and should not be uploaded unless you intentionally publish a separate release artifact.

A successful training run prints a `Round 100` record and a `Best: Round ...`
record. The fixed runners save the exact command and configuration, use a
`.done` marker only after validation, and can resume completed matrices. The
summary scripts produce CSV/Markdown or Excel files used to populate the main
and supplementary tables. Partition reports contain per-client totals,
client-class zero counts, the count-matrix checksum, and consistency checks.

## Tests

Run the Python functional tests from the repository root:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

On Linux or WSL, validate the experiment matrices without starting real
training:

```bash
bash tests/test_missing_class_plan.sh
bash tests/test_fedavg_vit_alpha_plan.sh
bash tests/test_backbone_rn50_plan.sh
bash tests/test_vit_minpc0_methods_plan.sh
```

## Citation

If you use this code, cite the corresponding FedProRef manuscript:

```bibtex
@article{wang2026fedproref,
  title  = {FedProRef: Communication-Efficient Prototype-Count Supervision for Federated Learning Under Severe Label Skew},
  author = {Wang, Lei and Ren, Hao},
  year   = {2026}
}
```
