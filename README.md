# FedProRef

**Prototype-Conditioned Feature Refinement for Federated Learning Under Severe Label Skew**

FedProRef is a feature-space federated learning (FL) framework for severe label skew. It converts compact cross-client prototype-count statistics into additional supervision for locally low-count classes in a shared feature space. In the evaluated implementation, clients cache features from a frozen OpenCLIP visual encoder and federate only a lightweight classifier head.

Each eligible client--class upload contains a prototype and its sample count, requiring \(\mathcal{O}(d+1)\) scalar values. The server forms class-wise anchor pools and trains a class-conditioned residual multilayer perceptron (MLP) to correct noisy prototype proposals. Selected clients then use the refined feature-label pairs during local classifier-head training before FedAvg aggregation.

This repository contains the experimental implementation and reproduction utilities used for the FedProRef manuscript. It runs the federated protocol as a single-machine simulation. Raw images, raw feature sets, and covariance matrices are not communicated by FedProRef.

## Evidence Scope

The manuscript evaluates a frozen-feature, head-only protocol using OpenCLIP ViT-B/16 features with \(d=512\). The formulation is defined over a generic shared feature space, but the reported experiments do not establish transfer to jointly updated encoders, heterogeneous client encoders, or other visual backbones.

The evaluated Dirichlet partitions enforce at least one local training example for every client--class pair. Therefore:

- the empirical results concern **locally low-count but observed classes**;
- the optional locally missing-class branch is not exercised by the reported experiments;
- no empirical claim is made about locally vacant classes.

The three-setting mechanism study also reports `DirectAnchorAug`, which samples reliable clean anchors directly without Gaussian proposal perturbation or a learned refiner. The results indicate that clean cross-client anchor supervision supplies most of the gain in those three settings, while the FedProRef--ProtoAug comparison supports the narrower role of the refiner as correction for noisy proposals. The comparison does not establish superiority or statistical equivalence between FedProRef and DirectAnchorAug.

## Method Summary

FedProRef follows this pipeline:

1. Extract and cache normalized features from a shared frozen visual encoder.
2. Partition the training split across clients using seeded Dirichlet label skew.
3. Before federated rounds, each client uploads eligible class prototypes and counts.
4. The server merges compatible same-class entries into class-wise anchor pools.
5. The server trains a class-conditioned residual MLP on Gaussian-perturbed anchor proposals.
6. Selected clients train classifier heads on real cached features and refined synthetic pairs for locally low-count classes.
7. The server aggregates classifier heads with FedAvg.
8. Every 10 rounds, the implementation continues refiner optimization on the same cached anchor pool.

Prototype refresh is disabled in the reported default experiments. Consequently, the periodic 50-epoch continuation steps do not incorporate new client statistics, track a changing encoder, or adapt to classifier-head updates. They continue optimization of the same server-side objective rather than forming a separately validated adaptive component.

The final method reported in the paper uses `--method fedproref --refiner_type mlp`. The RF-style refiner remains available as an auxiliary ablation through `--refiner_type rf`.

## Repository Layout

```text
.
├── config.py                              # command-line configuration
├── data_utils.py                          # datasets, partitions, feature caches
├── client_stats.py                        # client prototype-count statistics
├── server_calibration.py                  # anchor pools, refiner training, synthesis
├── direct_anchor_aug.py                   # DirectAnchorAug mechanism control
├── refiner_mlp.py                         # class-conditioned residual MLP
├── refiner_rf.py                          # auxiliary RF-style refiner
├── proposal.py                            # proposal and anchor sampling
├── head.py                                # classifier heads and FedAvg utilities
├── federated_loop.py                      # main simulation entry point
├── run.py                                 # repeated-run wrapper
├── run_experiment.sh                      # compact repeated-run launcher
├── summarize_results.py                   # log and result summarization
├── visualize.py                           # training-log visualization
├── train_plan_*.sh                        # paper experiment plans
├── configs/ablations/                     # declarative ablation metadata
├── scripts/                               # DirectAnchorAug preparation/run utilities
├── analysis_scripts/                      # post-hoc analysis utilities
├── analysis_plans/                        # documented analysis protocols
└── tests/                                 # unit tests
```

Datasets, pretrained weights, feature caches, training logs, generated results, and manuscript sources are not stored in this repository.

## Environment

Python 3.9 or later is recommended. Install the PyTorch build appropriate for the local CPU/CUDA environment first, followed by the remaining dependencies:

```bash
pip install torch torchvision
pip install open-clip-torch scikit-learn numpy pillow matplotlib openpyxl
```

GPU users should follow the official PyTorch installation instructions for their CUDA version.

Run the available unit tests with:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Data

The reported experiments use CIFAR-10, CIFAR-100, and Tiny-ImageNet. CIFAR-10 and CIFAR-100 are downloaded through `torchvision`. Tiny-ImageNet should be available as either:

```text
data/tiny-imagenet-200/
```

or:

```text
data/tiny-imagenet-200.zip
```

The Dirichlet split is generated by `dirichlet_partition` in `data_utils.py`. To reproduce the protocol-aligned nonempty client--class floor used in the paper, pass:

```text
--min_require_size 1
```

Training and test data remain separate: Dirichlet label skew is applied only to the training split.

## Pretrained Checkpoint

The paper experiments use OpenCLIP ViT-B/16 with feature dimension `d=512`:

```text
Backbone:   ViT-B-16
Tag:        openai
Identifier: timm/vit_base_patch16_clip_224.openai/open_clip_model.safetensors
SHA-256:    4b8699299b1e8997753c64b052ba32031449d5d853f55a039148560ee02b820f
```

The model can be loaded through OpenCLIP with:

```text
--backbone ViT-B-16 --pretrained openai
```

Alternatively, place a verified local checkpoint under `pretrain_path/` and pass its path through `--pretrained`. Pretrained weights are intentionally excluded from Git.

## Quick Smoke Test

The following command runs a short head-only FedAvg check:

```bash
python federated_loop.py \
  --method fedavg \
  --dataset cifar10 \
  --alpha 0.1 \
  --num_clients 10 \
  --select_clients 10 \
  --min_require_size 1 \
  --comm_rounds 5 \
  --local_epochs 1 \
  --backbone ViT-B-16 \
  --pretrained openai \
  --device auto \
  --exp_name quick_fedavg
```

This checks the pipeline but does not reproduce a reported paper result.

## Main FedProRef Protocol

The following command exposes the main CIFAR-100, \(\alpha=0.01\), partition-seed-42 configuration:

```bash
python federated_loop.py \
  --method fedproref \
  --refiner_type mlp \
  --dataset cifar100 \
  --alpha 0.01 \
  --num_clients 10 \
  --select_clients 10 \
  --min_require_size 1 \
  --comm_rounds 100 \
  --local_epochs 10 \
  --batch_size 64 \
  --lr 0.001 \
  --backbone ViT-B-16 \
  --pretrained openai \
  --feat_dim 512 \
  --head_type linear \
  --num_modes 1 \
  --proposal_sigma 0.05 \
  --gen_per_class 500 \
  --aug_gen_per_class 100 \
  --weak_class_percentile 10 \
  --min_samples_per_class 10 \
  --refiner_pretrain_epochs 300 \
  --refiner_finetune_epochs 50 \
  --refiner_lr 0.001 \
  --cal_every 10 \
  --w_proto 0.1 \
  --w_reg 0.01 \
  --seed 42 \
  --partition_seed 42 \
  --device auto \
  --exp_name fedproref_cifar100_a001_s42
```

Repeat with training seeds 42, 43, and 44 while keeping `--partition_seed 42` fixed to obtain the paper's three-seed main-protocol design.

The repeated-run wrapper automates that seed progression:

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

> **Warning:** the current `run.py` wrapper recreates `./checkpoints` when it starts. Back up checkpoints that must be retained before invoking the wrapper.

## DirectAnchorAug Mechanism Control

`DirectAnchorAug` is exposed through:

```text
--method direct_anchor_aug --refiner_type none
```

It uses the same active merged anchor pool and local synthetic-sample budget as the matched client-side augmentation pipeline, samples anchors according to their counts, and applies neither Gaussian proposal noise nor a learned refiner.

Example for the CIFAR-100, \(\alpha=0.01\), seed-42 setting:

```bash
python federated_loop.py \
  --method direct_anchor_aug \
  --refiner_type none \
  --dataset cifar100 \
  --alpha 0.01 \
  --num_clients 10 \
  --select_clients 10 \
  --min_require_size 1 \
  --min_samples_per_class 10 \
  --backbone ViT-B-16 \
  --pretrained openai \
  --feat_dim 512 \
  --head_type linear \
  --comm_rounds 100 \
  --local_epochs 10 \
  --batch_size 64 \
  --lr 0.001 \
  --num_modes 1 \
  --proto_merge_threshold 0.90 \
  --proposal_sigma 0.05 \
  --gen_per_class 500 \
  --aug_gen_per_class 100 \
  --weak_class_percentile 10 \
  --seed 42 \
  --partition_seed 42 \
  --device auto \
  --exp_name direct_anchor_aug_cifar100_a001_s42
```

`proposal_sigma` remains part of the shared command-line schema but is not used to perturb DirectAnchorAug samples. The declarative matched configuration is recorded in `configs/ablations/direct_anchor_aug.yaml`, and the isolated sampling path is covered by `tests/test_direct_anchor_aug.py`.

## Experiment Plans

The repository includes the experiment-plan scripts used for the main comparisons and focused studies:

| Script | Purpose |
| --- | --- |
| `train_plan_cifar_tiny.sh` | Main CIFAR/Tiny-ImageNet method and alpha sweeps |
| `train_plan_cifar100_a001_seed_matrix.sh` | CIFAR-100 partition/training-seed matrix |
| `train_plan_tc50_sc10_seed42_43_44.sh` | 50-client, 10-selected-client stress protocol |
| `train_plan_fedproref_sensitivity.sh` | Reported sensitivity settings |
| `train_plan_fedproref_refnone.sh` | No-refiner ablation |
| `train_plan_fedproref_refrf.sh` | RF-style refiner ablation |
| `train_plan_fedproref_refmlp_nosim.sh` | No-similarity-merge ablation |
| `train_plan_weak_class_mechanism_minimal.sh` | Focused weak-class mechanism runs |

Several shell scripts contain machine-specific default values for `PYTHON_BIN` and a local checkpoint path. Override them when running on another system, for example:

```bash
PYTHON_BIN=python \
PRETRAINED=openai \
METHODS="fedproref proto_aug proto_cal" \
bash train_plan_cifar_tiny.sh
```

Always inspect a plan before launching a full sweep. Where supported, use `DRY_RUN=1` to print commands without training.

## 50-Client Stress-Test Protocol

For the joint client-scale and partial-participation protocol, use:

```text
--num_clients 50 --select_clients 10 --min_require_size 1
```

The implementation samples 10 distinct clients uniformly without replacement in each communication round. All compared methods follow this stochastic rule, but realized round-wise client sets are not constrained to match across methods. The manuscript therefore treats the three-seed TC50 results as a descriptive, non-paired comparison rather than a client-schedule-paired experiment.

## Methods and Ablations

| Command | Role |
| --- | --- |
| `--method fedavg` | Head-only FedAvg with frozen cached features. |
| `--method proto_cal` | Controlled prototype-mean server-calibration baseline. |
| `--method proto_aug` | Client-side low-count-class augmentation with unrefined Gaussian proposals. |
| `--method proto_sample` | Server-side calibration using Gaussian-sampled prototype features. |
| `--method direct_anchor_aug --refiner_type none` | Clean-anchor client-side mechanism control. |
| `--method fedproref --refiner_type mlp` | Main FedProRef configuration. |
| `--method fedproref --refiner_type rf` | Auxiliary RF-style refiner ablation. |
| `--method fedproref --refiner_type mlp --no_proto_similarity_merge` | FedProRef without same-class prototype similarity merging. |

## Main Experimental Settings

| Item | Paper setting |
| --- | --- |
| Feature extractor | OpenCLIP ViT-B/16 |
| Feature dimension | 512 |
| Feature handling | Cached and L2-normalized after `encode_image()` |
| Classifier head | `Linear(512, C)` |
| Main client protocol | 10 total; all 10 selected per round |
| Stress client protocol | 50 total; 10 selected per round |
| Communication rounds | 100 |
| Local epochs | 10 |
| Batch size | 64 |
| Local optimizer | Adam, learning rate `1e-3`, weight decay 0 |
| Non-IID split | Dirichlet label skew on the training split |
| Main partition seed | 42 |
| Training seeds | 42, 43, 44 |
| Main alpha values | 0.01, 0.03, 0.05 |
| Extended alpha values | 0.01, 0.03, 0.05, 0.07, 0.09, 0.10, 0.30, 0.50 |
| Partition floor | `min_require_size = 1` |

FedProRef-specific settings:

| Item | Paper setting |
| --- | --- |
| Refiner | Class-conditioned residual MLP |
| Class embedding dimension | 512 |
| Architecture | `Linear(1024,512)-ReLU-Linear(512,512)-ReLU-Linear(512,512)` |
| Refined feature | `Normalize(z0 + r_phi(z0, e_c))` |
| Refiner optimizer | Adam, learning rate `1e-3` |
| Initial / continuation optimization | 300 epochs / 50 epochs |
| Continuation period | Every 10 communication rounds |
| Proposal noise | `sigma = 0.05` |
| Server proposals per class | `gen_per_class = 500` |
| Synthetic pairs per low-count class | `aug_gen_per_class = 100` |
| Low-count thresholds | Absolute count `< 10`; bottom 10 percentile |
| Prototype merge threshold | Initial 0.90; adaptive range [0.70, 0.98] |
| Loss weights | `lambda_proto = 0.1`, `lambda_reg = 0.01` |

## Communication-Claim Boundary

FedProRef uploads one \(d\)-dimensional prototype and one count for each eligible client--class record, compared with a class mean and full covariance matrix in the component-level reference calculation. At \(d=512\), this corresponds to approximately 512 times fewer uploaded scalar values per eligible class-statistics record.

This is a **client-to-server, component-level class-statistics comparison**. It is not a claim that FedProRef reduces complete bidirectional communication by 512 times. Shared classifier-head traffic and method-specific downstream transmissions must be considered separately.

## Outputs

Runtime artifacts are written to:

```text
logs/          # training logs
checkpoints/   # saved classifier heads
results/       # run metadata, summaries, and spreadsheets
data/*_cache/  # extracted feature and partition caches
```

These paths and large model/data artifacts are excluded by `.gitignore` and should be distributed separately when needed.

## Citation

If you use this code, please cite the FedProRef manuscript:

```bibtex
@misc{wang2026fedproref,
  title  = {FedProRef: Prototype-Conditioned Feature Refinement for Federated Learning Under Severe Label Skew},
  author = {Wang, Lei and Ren, Hao},
  year   = {2026},
  note   = {Manuscript submitted to IEEE Access}
}
```

Update this citation after the article receives final publication metadata.
