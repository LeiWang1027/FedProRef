# Analysis 1: Synthetic Feature Distribution Evidence

## Goal

Answer whether the MLP-refined synthetic features in the CIFAR-100 alpha=0.01 seed-42 main FedProRef run:

1. are more class-consistent than naive Gaussian prototype proposals;
2. do not collapse onto prototype anchors;
3. preserve non-trivial within-class diversity.

This analysis is offline. It must not rerun federated training, must not silently regenerate missing artifacts under a different setting, and must not change any existing experimental numbers.

## Required Inputs

Stop immediately and report missing inputs if any item below is absent.

1. CIFAR-100 alpha=0.01 seed-42 main-run feature cache:
   - full training features;
   - full training labels;
   - client indices for the same partition.

2. Final deployed MLP refiner checkpoint from the same run:
   - final refiner parameters phi^T after 100 communication rounds;
   - class embedding parameters;
   - architecture metadata sufficient to reconstruct the refiner.

3. Merged prototype pool actually used by the main experiment:
   - class-wise prototype anchors after the same prototype aggregation/merge logic;
   - per-anchor counts;
   - merge threshold/configuration metadata.

The exact target setting is:

| Field | Required value |
| --- | --- |
| Dataset | CIFAR-100 |
| Dirichlet alpha | 0.01 |
| Training seed | 42 |
| Partition seed | 42 |
| Method | FedProRef main method |
| Refiner | MLP |
| Proposal sigma | 0.05 |
| Synthetic number per class | M = 100 |
| Number of classes | 100 |

## Artifact Audit

Before computing anything, the script must write an audit JSON under:

```text
results/analysis1_synthetic_distribution/artifact_audit.json
```

The audit must include:

- absolute paths of all loaded artifacts;
- feature cache keys and array shapes;
- checkpoint key list or structured metadata;
- prototype-pool class coverage;
- SHA256 hashes or file mtimes/sizes for traceability;
- a boolean `can_run_analysis`.

If `can_run_analysis=false`, the script exits with a non-zero status and writes the missing reason. It must not create placeholder metrics.

## Objects To Generate

Generate objects for all 100 CIFAR-100 classes. Do not select representative classes only.

For each class `c`:

1. Reference real features:
   - all cached real training features with label `c`;
   - L2-normalized as stored/used by the main pipeline.

2. Reference prototype anchors:
   - all merged prototype anchors in `P_c`;
   - each anchor has count `n`.

3. Naive group:
   - sample M = 100 anchors from `P_c`;
   - sample anchors with probability `n / sum_j n_j`, matching Eq. (5);
   - use fixed paired noise epsilon;
   - compute `z0 = Normalize(mu + sigma * epsilon)`, sigma = 0.05.

4. Refined group:
   - feed the exact same `z0` into the final MLP refiner;
   - compute `z_hat = Normalize(z0 + r_phi(z0, e_c))`;
   - use the same generated proposal/noise pairing as the naive group.

All generation must be deterministic with an explicit analysis seed, e.g. `analysis_seed=20260709`, recorded in the output metadata.

## Metrics

Compute per-class metrics first, then report mean +/- std across the 100 classes.

### A. Class Consistency

Purpose: verify refined features are closer to the real class structure than naive proposals.

For each class:

- real centroid: normalized mean of real training features of class `c`;
- `cos_to_real_centroid_naive`;
- `cos_to_real_centroid_refined`;
- paired difference: `refined - naive`;
- optional paired sign/Wilcoxon test across classes.

Expected evidence pattern:

- refined mean cosine to real centroid should be higher than naive;
- report the actual value, not just the sign.

### B. Prototype Non-Collapse

Purpose: verify refined features are not merely prototype means.

For each class:

- nearest same-class prototype cosine for naive samples;
- nearest same-class prototype cosine for refined samples;
- average Euclidean distance to the sampled source anchor for naive;
- average Euclidean distance to the sampled source anchor for refined.

Interpretation:

- if refined samples have cosine nearly 1.0 to anchors and almost zero distance, they collapsed;
- if refined samples improve class consistency while retaining measurable distance from anchors, they are not equivalent to ProtoCal.

### C. Synthetic Diversity

Purpose: verify M = 100 samples do not collapse into one point.

For each class and for naive/refined separately:

- average pairwise cosine distance among the 100 synthetic samples;
- effective rank of the centered synthetic feature matrix;
- mean distance to the class synthetic centroid.

Interpretation:

- refined diversity can be lower than naive if refinement denoises proposals;
- it should not approach zero across most classes.

### D. Optional Real-Relative Diversity

Purpose: calibrate synthetic diversity against real feature spread.

For each class:

- average pairwise cosine distance on a capped real subset, e.g. up to 100 real features;
- ratio `diversity_refined / diversity_real`;
- ratio `diversity_naive / diversity_real`.

This metric is optional only if some classes lack enough real samples. If skipped, record why.

## Output Files

All outputs go under:

```text
results/analysis1_synthetic_distribution/
```

Required files:

```text
artifact_audit.json
per_class_metrics.csv
summary_metrics.json
generation_metadata.json
```

Optional files:

```text
paired_metric_tests.json
sampled_features_summary.npz
```

Do not save all synthetic feature arrays by default unless explicitly needed; save summaries first to keep the artifact lightweight.

## Pass/Fail Interpretation

This analysis supports the anti-collapse claim only if all three conditions are empirically supported:

1. Class consistency:
   - mean refined cosine to real centroid > mean naive cosine to real centroid.

2. Non-collapse:
   - refined samples are not almost identical to nearest prototype anchors;
   - report threshold-free values, and flag possible collapse if mean nearest-prototype cosine is extremely high, e.g. > 0.995.

3. Diversity:
   - refined average pairwise cosine distance is clearly above zero;
   - effective rank is not approximately 1 for most classes.

If one condition fails, report it directly. Do not reinterpret failed evidence as support.

## Planned Script

Create a script:

```text
analysis_scripts/analysis1_synthetic_feature_distribution.py
```

Suggested command:

```bash
/home/cherry/miniconda3/envs/fedfm/bin/python analysis_scripts/analysis1_synthetic_feature_distribution.py \
  --dataset cifar100 \
  --alpha 0.01 \
  --seed 42 \
  --partition_seed 42 \
  --feature-cache data/cifar100_clip_cache/ViT-B-16_old_open_clip_model.safetensors_alpha0.01_c10_s42.npz \
  --refiner-checkpoint checkpoints/<final_fedproref_mlp_refiner_phiT>.pth \
  --prototype-pool results/<main_run_merged_prototype_pool>.pt \
  --sigma 0.05 \
  --samples-per-class 100 \
  --analysis-seed 20260709 \
  --out-dir results/analysis1_synthetic_distribution
```

The placeholder checkpoint and prototype-pool paths must be replaced with verified artifacts. The script should not guess them.

## Known Risk In Current Project

The existing classifier checkpoints may contain only classifier-head weights (`fc.weight`, `fc.bias`). If the final MLP refiner state and merged prototype pool were not saved during the main FedProRef run, this analysis cannot be computed from the current artifacts.

In that case, the correct outcome is:

1. write `artifact_audit.json` with `can_run_analysis=false`;
2. report missing final MLP refiner checkpoint and/or prototype pool;
3. stop without metrics, figures, or manuscript changes.

Producing these artifacts later would require changing training-time saving logic and rerunning the exact main setting. That rerun is outside this offline analysis plan unless explicitly requested.



## Implementation Status

Implemented files:

```text
analysis_scripts/analysis1_synthetic_feature_distribution.py
```

Training-time artifact saving has also been added to `federated_loop.py`. Future FedProRef runs with an active refiner will still save the original classifier-head checkpoint, and will additionally save:

```text
checkpoints/<exp_name>_<method>_analysis_artifacts.pt
```

The artifact contains:

- `refiner_state_dict`, including class embeddings;
- `prototype_pool`, the merged prototype pool used by the run;
- `client_stats`;
- `proto_merge_threshold`;
- `args`;
- `last_refiner_train_metrics`.

Current audit result on the existing workspace:

- Required CIFAR-100 alpha=0.01 seed-42 feature cache exists.
- Existing `refmlp` checkpoints are classifier-head checkpoints only (`fc.weight`, `fc.bias`).
- No saved final MLP refiner checkpoint exists for the completed main run.
- No saved merged prototype pool exists for the completed main run.

Therefore the current workspace still cannot compute the requested metrics until the exact main setting is rerun after the artifact-saving change.

Audit-only command already tested:

```bash
/home/cherry/miniconda3/envs/fedfm/bin/python analysis_scripts/analysis1_synthetic_feature_distribution.py \
  --dataset cifar100 \
  --alpha 0.01 \
  --seed 42 \
  --partition_seed 42 \
  --feature-cache data/cifar100_clip_cache/ViT-B-16_old_open_clip_model.safetensors_alpha0.01_c10_s42.npz \
  --refiner-checkpoint checkpoints/plan_fedproflow_cifar100_a0.01_tc10_sc10_le10_ps42_s42_refmlp_fedproflow.pth \
  --prototype-pool results/missing_main_run_merged_prototype_pool.pt \
  --sigma 0.05 \
  --samples-per-class 100 \
  --analysis-seed 20260709 \
  --out-dir results/analysis1_synthetic_distribution_audit_only
```

It correctly stopped without metrics because the merged prototype pool is missing.

After rerunning the exact main setting with the updated code, use the combined artifact path:

```bash
/home/cherry/miniconda3/envs/fedfm/bin/python analysis_scripts/analysis1_synthetic_feature_distribution.py \
  --dataset cifar100 \
  --alpha 0.01 \
  --seed 42 \
  --partition_seed 42 \
  --feature-cache data/cifar100_clip_cache/ViT-B-16_old_open_clip_model.safetensors_alpha0.01_c10_s42.npz \
  --artifact checkpoints/<exp_name>_fedproref_analysis_artifacts.pt \
  --sigma 0.05 \
  --samples-per-class 100 \
  --analysis-seed 20260709 \
  --out-dir results/analysis1_synthetic_distribution
```
