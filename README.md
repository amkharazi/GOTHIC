# Fast Start

Checkout `run.sh` for ready-to-run benchmark commands.

> Example (Real dataset / **Wine** + **GOTHIC**):
>
> ```bash
> python runner.py --model gothic --run_id wine_ID001 --dataset wine --seed 42 --train_fraction 0.8 --split_strategy balanced --n_micro 5 --kmeans_n_init 50 --kmeans_max_iter 500 --dist_quantile 0.6 --b_boundary 5 --d_model 128 --n_heads 2 --attn_hidden 64 --n_transformer_layers 3 --train_epochs 2000 --lr 0.003 --weight_decay 0.001 --k_target 3 --data_root datasets --out_dir outputs/gothic/real
> ```

Windows reminder (Git Bash):

```text
"C:\Program Files\Git\usr\bin\bash.exe" -lc "cd '/c/Users/Snapp! Pay/Desktop/GOTHIC/GOTHIC' && bash run.sh"
```

---

# GOTHIC: Graph-Overclustered Transformer-based Hierarchical Integrated Clustering

GOTHIC is an experimental clustering framework that combines:

- **Over-clustering** (KMeans → many micro-clusters),
- A **graph** over micro-clusters (edges based on centroid + boundary + density),
- A **Transformer-based pair scorer** over that graph,
- **Hierarchical merging** of micro-clusters guided by learned pairwise scores.

Acronym:

> **G**raph-**O**verclustered **T**ransformer-based **H**ierarchical **I**ntegrated **C**lustering

This repo also contains **baseline models** (classical + deep) so you can benchmark GOTHIC side-by-side on both synthetic and real datasets.

---

## Supported models (`runner.py --model ...`)

Common pattern:

```bash
python runner.py --model <model_name> --run_id <ID> --dataset <dataset_name> ...other args...
```

Currently supported (based on the latest benchmark runs):

- `gothic` — Graph over micro-clusters + Transformer pair scorer + hierarchical merging (main method)
- `kmeans` — KMeans baseline
- `dbscan` — DBSCAN baseline
- `hdbscan` — HDBSCAN baseline
- `insdpc` — INSDPC baseline
- `amd_dbscan` — AMD-DBSCAN baseline
- `mdbscan` — MDBSCAN baseline
- `spectral` — Spectral clustering baseline
- `idec` — Improved Deep Embedded Clustering baseline
- `gnn` — GNN clustering baseline
- `gnn_contrastive` — contrastive GNN clustering variant

> Note: different models accept different hyperparameters, but they share the same **experiment management** (`--dataset`, `--seed`, `--run_id`, `--data_root`, `--out_dir`) and produce a consistent **results.txt + figures** layout.

---

## Supported datasets

### Synthetic (`datasets/synthetic/`)

SIPU-style 2D datasets (downloaded as `.txt`):

- `compound` → `Compound.txt`
- `aggregation` → `Aggregation.txt`
- `d31` → `D31.txt`
- `flame` → `flame.txt`
- `jain` → `jain.txt`
- `pathbased` → `pathbased.txt`
- `r15` → `R15.txt`

Plus:

- `noisy_circles` → generated via `sklearn.make_circles`, saved as `noisy_circles.csv` (`x1, x2, label`)

### Real (`datasets/real/`)

CSV datasets (features + final `label` column):

- `breast_cancer`
- `iris`
- `wine`
- `digits` (sklearn digits)
- `olivetti_faces` (sklearn Olivetti faces)

---

## High-dimensional datasets (PCA / feature caps)

Some datasets (e.g., **digits**, **olivetti_faces**) are high-dimensional. Recent updates add optional dimensionality control so runs remain stable and fast:

- `--reduce_dim {none,pca}`  
  - `none`: use original features  
  - `pca`: reduce to `--pca_dim` components before graph + density features
- `--pca_trigger_dim <int>`  
  If the original dimensionality `d` is above this threshold, PCA reduction can be enabled automatically (depending on your configuration).
- `--pca_dim <int>`  
  PCA output dimensionality (common value: `128`).
- `--max_density_dim <int>`  
  Caps the dimensionality used specifically in density-related computations (common value: `64`).

These settings are logged per run (e.g., `reduce_dim`, `used_dim`) so your summaries remain reproducible.

---

## High-level idea (GOTHIC)

Given a dataset \(X \in \mathbb{R}^{N \times d}\) with labels \(y\) (used **only** for supervision at the micro-cluster level):

1. **Over-cluster with KMeans**
   - Choose `n_micro`.
   - Run KMeans on the **train** subset only.
   - Each micro-cluster has a centroid, boundary points, and a majority label.

2. **Build a micro-cluster graph**
   - Nodes: micro-clusters.
   - Edges: connect pairs whose centroid distance is below a quantile threshold `dist_quantile`.
   - Edge features include centroid distance, boundary distance, and density-based features.

3. **Micro-cluster node features**
   - Centroid coordinates + size/radius/density summaries.

4. **Transformer-based pair scorer**
   - Transformer encoder processes node features.
   - MLP predicts edge merge probability \(P(\text{merge})\) for each graph edge.
   - Best epoch is selected by training performance.

5. **Hierarchical merging**
   - Greedy Union-Find merging along highest \(P(\text{merge})\) edges until reaching `k_target`.
   - If the graph is too sparse, fallback merges are distance-based.

6. **Assign labels to points**
   - Train points map micro → macro clusters.
   - Test points map to nearest effective micro-cluster then macro cluster.

7. **Evaluation & visualization**
   - External metrics: NMI / AMI / ARI / FMI / ACC (Hungarian)
   - Internal metrics: Silhouette / Davies–Bouldin / Calinski–Harabasz
   - Diagnostic plots are saved as **transparent PNG** under each run’s `figures/` folder.

---

## Project structure (current)

```text
gothic_project/
  helper.py
  dataset.py
  loader.py

  gothic_model.py

  # Baseline models
  kmeans_model.py
  dbscan_model.py
  hdbscan_model.py
  insdpc_model.py
  amd_dbscan_model.py
  mdbscan_model.py
  spectral_model.py
  idec_model.py
  gnn_model.py
  gnn_contrastive_model.py

  runner.py
  run.sh
  requirements.txt

  datasets/
    synthetic/
    real/

  outputs/
    <model_name>/
      synthetic/
        <run_id>/
          checkpoints/
          figures/
          <dataset>_results.txt
      real/
        <run_id>/
          checkpoints/
          figures/
          <dataset>_results.txt
```

> Your `--out_dir` can point anywhere, but the recent convention is:
>
> - `outputs/<model_name>/synthetic` for synthetic datasets
> - `outputs/<model_name>/real` for real datasets

---

## Installation

```bash
pip install -r requirements.txt
```

Python 3.9+ is expected (3.11 recommended).

---

## `dataset.py` – downloading / generating datasets

```bash
python dataset.py                       # download/generate ALL supported datasets
python dataset.py --datasets compound   # only Compound
python dataset.py --datasets compound aggregation noisy_circles iris
```

Programmatic:

```python
from dataset import ensure_dataset_exists
path = ensure_dataset_exists("compound")
```

---

## `runner.py` – benchmark entry point

`runner.py` is the generic entry point that selects a model via `--model` and forwards the remaining CLI arguments.

Example (synthetic / Compound):

```bash
python runner.py --model gothic --run_id compound_ID001 --dataset compound --seed 42 --train_fraction 0.8 --split_strategy balanced --n_micro 91 --kmeans_n_init 50 --kmeans_max_iter 500 --dist_quantile 0.3 --b_boundary 5 --d_model 32 --n_heads 4 --attn_hidden 64 --n_transformer_layers 2 --train_epochs 2000 --lr 0.001 --weight_decay 0.0001 --k_target 6 --data_root datasets --out_dir outputs/gothic/synthetic
```

---

## Outputs

Each run writes:

- `<out_dir>/<run_id>/<dataset>_results.txt` — metrics summary for train/test
- `<out_dir>/<run_id>/figures/*.png` — transparent diagnostic figures
- `<out_dir>/<run_id>/checkpoints/*` — saved model checkpoints (when applicable)

Recent convention examples:

```text
outputs/gothic/real/wine_ID001/wine_results.txt
outputs/gothic/real/wine_ID001/figures/wine_metric_acc.png
```

---

## Result summaries (`results_summary.csv`)

Recent benchmark runs produce a consolidated CSV (example: `results_summary.csv`) containing:

- dataset, model, run_id, seed
- key hyperparameters (where applicable)
- train/test metrics (NMI/AMI/ARI/FMI/ACC + internal metrics)

Quick “best per dataset” snippet:

```python
import pandas as pd

df = pd.read_csv("results_summary.csv")

best = (
    df.sort_values("test_acc", ascending=False)
      .groupby("dataset", as_index=False)
      .first()[["dataset", "model", "test_acc", "test_nmi", "test_ari"]]
)

print(best)
```

---

## Ablation studies (seeds + one-factor sweeps)

To measure stability and sensitivity (confidence intervals, variance, plots), use an ablation script like `ablation_study.py`:

- **Seed sweep:** same config across multiple seeds → mean/std/var + plots
- **One-factor sweeps (seed fixed):** vary one hyperparameter at a time (e.g., `n_micro`, `dist_quantile`, Transformer depth)

Typical usage:

```bash
python ablation_study.py                 # runs seed sweep + ablations (default)
python ablation_study.py --only_seed_sweep
python ablation_study.py --only_ablations
python ablation_study.py --no_skip_existing
```

Outputs are saved under:

```text
<out_dir>/ABlationReports/<dataset>/<timestamp>/
  seed_sweep_raw.csv
  seed_sweep_stats.csv
  ablations_raw.csv
  plots/*.png   # transparent
```

---

## Reproducibility notes

- `--seed` controls random seeds for Python / NumPy / (PyTorch when used).
- Keep dataset + all CLI args + library versions fixed for exact reproducibility.
- `--run_id` is intended as an experiment ID; use unique IDs per run to avoid overwriting outputs.
