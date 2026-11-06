# TCAP

[English](README.md) | [简体中文](README_zh-CN.md)

**Target-Conditioned Asymmetric Parameter Projection** selects task checkpoints for tabular classification. After adapting a shared base model to the target task once, TCAP ranks source checkpoints using their parameter displacements. Use the top-1 checkpoint directly, or optionally select among the top-k candidates by validation log-loss.

Supports **TabICL-v2** and **TabPFN-v3**. **Built with PriorLabs-TabPFN.**

## Installation

Requires Python 3.10+ and PyTorch. Install the backend you need from the repository root:

```bash
python -m pip install ".[tabicl]"
# Or use TabPFN:
python -m pip install ".[tabpfn]"
```

Prepare your own classification CSVs and base checkpoint. Backend packages are pinned to `tabicl==2.1.1` and `tabpfn==8.5.0`; use the corresponding v2/v3 classifier checkpoint. To use an installed source version of a backend, install TCAP with `python -m pip install .`.

## Quick Start

Each task uses one CSV. The label column defaults to `y`, or the last column if `y` is absent; use `--label COLUMN` to specify another column. Supports up to 10 classes.

### 1. Adapt tasks

Adapt each source and the target from the same base checkpoint using identical settings. The following example uses Bash:

```bash
BASE=/path/to/tabicl-classifier-v2-20260212.ckpt

for task in source_a source_b source_c source_d source_e target; do
  tcap adapt \
    --backend tabicl \
    --base "$BASE" \
    --data "data/${task}.csv" \
    --task-key "$task" \
    --output "artifacts/tabicl/${task}"
done
```

Each task directory contains a checkpoint, `delta.npy`, and `task.json`. Use a fresh output directory for each run.

For TabPFN-v3, adapt all tasks with `--backend tabpfn` and a shared TabPFN-v3 classifier checkpoint, then create a separate bank of their artifacts.

### 2. Create a checkpoint bank

Save this as `bank.json`. Paths are relative to the bank file:

```json
[
  "artifacts/tabicl/source_a/task.json",
  "artifacts/tabicl/source_b/task.json",
  "artifacts/tabicl/source_c/task.json",
  "artifacts/tabicl/source_d/task.json",
  "artifacts/tabicl/source_e/task.json"
]
```

### 3. Predict

Default: select the highest-scoring external checkpoint and predict once.

```bash
tcap predict \
  --target artifacts/tabicl/target/task.json \
  --bank bank.json \
  --data data/target.csv \
  --output predictions/top1
```

Optional: retrieve five candidates and select by validation log-loss.

```bash
tcap predict \
  --target artifacts/tabicl/target/task.json \
  --bank bank.json \
  --data data/target.csv \
  --rerank --top-k 5 \
  --output predictions/top5_val
```

Use the same target CSV as adaptation. Prediction runs on its held-out query rows. Results are saved as `selection.json` and `predictions.npz`; the latter contains `probabilities`, `classes`, `predictions`, and zero-based `row_indices`.

## Selection Options

| Option | Behavior |
|---|---|
| Default | Top-1 selection, excluding the target self-checkpoint |
| `--rerank --top-k K` | Validate the top-k candidates; default `k=5` |
| `--include-self` | Allow the target self-checkpoint in direct mode |
| `--no-include-self` | Exclude the target self-checkpoint in reranking mode |
| `--mu 0.1` | Module-weight smoothing |
| `--tiny-energy-fallback` | Enable uniform weights for total target module energy <= `1e-12`; disabled by default |
| `--n-estimators 8` | Estimators for validation and final prediction |
| `--device cuda` | Execution device; use `cpu` for CPU execution |

Reranking includes the target self-checkpoint in the candidate pool by default. It competes by TCAP score and is not forced into top-k. Selection uses validation log-loss, then validation accuracy and TCAP rank to break ties. At least k eligible candidates are required.

## Method and Defaults

Task displacement is `delta_i = theta_i - theta_0`. Module weights are computed from target mean squared displacements and smoothed with `mu=0.1`. TCAP scores candidates by a weighted projection:

```text
energy[m] = ||delta_target[m]||² / dim[m]
weight[m] = (1 - mu) * energy[m] / sum(energy) + mu / number_of_modules
score = sum_m weight[m] * <delta_target[m], delta_source[m]>
        / (sqrt(sum_m weight[m] * ||delta_target[m]||²) + 1e-12)
```

The source norm is not used for normalization. TabICL-v2 uses 15 parameter modules; TabPFN-v3 uses 34. Candidates must share the same backend, base checkpoint, and adaptation settings.

By default, every positive total module energy is normalized by the formula above, including values <= `1e-12`. Zero total energy raises an error because the weight normalization is undefined. To opt in to uniform weights when total energy <= `1e-12` (including zero), pass `--tiny-energy-fallback` to `tcap predict`. The Python interfaces `target_weights`, `tcap_score`, `rank_candidates`, and `predict_task` accept `tiny_energy_fallback=True`; the default is `False`. `selection.json` records this option. With fallback enabled, a zero target displacement gives zero scores, with ties resolved by task key.

| Setting | Default |
|---|---|
| Stratified split | 64% training / 8% early stopping / 8% reranking / 20% queries |
| Context | 72% for reranking; 80% for final prediction |
| Optimizer | AdamW, learning rate `1e-5`, weight decay `0.01` |
| Training | Up to 30 epochs, early-stopping ROC-AUC, patience 8 |
| Batch size | 4; restart at 2, then 1 after CUDA OOM |
| Estimator views | 32 for training and early stopping; 8 for prediction |
| Seed | 42 |

The four subsets are disjoint. Held-out query labels are used only to stratify the split and are not passed to checkpoint selection or prediction. Each subset must contain every class.

## Testing

```bash
python -m pip install ".[test]"
python -m pytest -q
```

## Acknowledgements

Built on [TabICL](https://github.com/soda-inria/tabicl) and [TabPFN](https://github.com/PriorLabs/TabPFN). See [third-party notices](THIRD_PARTY_NOTICES.md) and [licenses/](licenses/) for attribution and upstream licenses.
