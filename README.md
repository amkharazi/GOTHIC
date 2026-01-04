# Fast Start

Checkout `run.sh` for the runner scripts. 

> For example, you can use this command to run the script on the *Compound* dataset with the GOTHIC model:
>
> ```bash
> python runner.py --model gothic --run_id compound_ID001 --dataset compound --seed 42 --train_fraction 0.8 --split_strategy balanced --n_micro 91 --kmeans_n_init 50 --kmeans_max_iter 500 --dist_quantile 0.3 --b_boundary 5 --d_model 32 --n_heads 4 --attn_hidden 64 --n_transformer_layers 2 --train_epochs 2000 --lr 0.001 --weight_decay 0.0001 --k_target 6 --data_root datasets --out_dir outputs
> ```

reminder: `"C:\Program Files\Git\usr\bin\bash.exe" -lc "cd /c/Git/GOTHIC && sed -i 's/\r$//' run.sh && bash run.sh"`
---

# GOTHIC: Graph-Overclustered Transformer-based Hierarchical Integrated Clustering

GOTHIC is an experimental clustering framework that combines:

- **Over-clustering** (KMeans → many micro-clusters),
- A **graph** over micro-clusters (edges based on centroid + boundary + density),
- A **Transformer-based pair scorer** over that graph,
- **Hierarchical merging** of micro-clusters guided by the learned pairwise scores.

The acronym stands for:

> **G**raph-**O**verclustered **T**ransformer-based **H**ierarchical **I**ntegrated **C**lustering

It’s designed to explore graph + neural approaches to clustering on both synthetic and real datasets, with a rich set of evaluation metrics and diagnostic plots.

---

## High-level idea

Given a dataset \(X \in \mathbb{R}^{N 	imes d}\) with labels \(y\) (used **only** for supervision at the micro-cluster level):

1. **Over-cluster with KMeans**  
   - Choose `n_micro` (e.g. 80 for Compound).  
   - Run KMeans on the **train** subset only.  
   - Each micro-cluster has:
     - A centroid,
     - Members and their distances to the centroid,
     - A majority label (from `y_train`) for supervision,
     - A set of **boundary points** (farthest members).

2. **Build a micro-cluster graph**
   - Nodes: micro-clusters.
   - Edges: connect pairs of micro-clusters whose centroid distance is below a quantile threshold `dist_quantile` (e.g. 0.3).
   - For each edge \((i, j)\), we compute **pairwise features**:
     - `center_dist`: distance between centroids,
     - `border_min_dist`: minimum distance between boundary points of the two micro-clusters,
     - `log_density_diff`: absolute difference between log-densities,
     - `log_density_i`, `log_density_j`: the (log) densities of the two micro-clusters themselves.

3. **Node (micro-cluster) features**
   - For each micro-cluster, we build a feature vector:
     - Centroid coordinates (dimension \(d\)),
     - `log_size = log(1 + |C_i|)`,
     - `radius_mean`,
     - `radius_std`,
     - `log_density` (size divided by a simple volume proxy, then log).
   - These density-aware features are what the Transformer sees as node inputs.

4. **Transformer-based pair scorer**
   - A small Transformer encoder is applied to all node features (micro-clusters) at once.
   - For each edge (pair of nodes) we build a **symmetric pair embedding**:
     - \([h_i, h_j, |h_i - h_j|, h_i \odot h_j]\) concatenated with the pairwise extra features  
       (`center_dist`, `border_min_dist`, `log_density_diff`, `log_density_i`, `log_density_j`).
   - A small MLP outputs a logit for “should these two micro-clusters belong to the same macro-cluster?”  
   - Training label for each edge: 1 if micro-cluster majority labels match, else 0.
   - We keep the **best epoch** (highest training accuracy, tie-broken by lower loss).

5. **Hierarchical merging**
   - Start with each micro-cluster as its own component (Union-Find).
   - Sort edges by **decreasing** predicted probability \(P(	ext{merge})\).
   - Merge components greedily along those edges until the number of components reaches `k_target`.
   - If we still have more than `k_target` components (graph too sparse), we fall back to **distance-based** merges.

6. **Assigning labels to points**
   - **Train points**: each point belongs to a micro-cluster → map it to its macro-cluster.
   - **Test points**: assign each to the nearest **effective** micro-cluster (the ones that actually appear after removing empty ones), then map that micro-cluster to a macro-cluster.

7. **Evaluation & visualization**

   For **train** and **test**, we compute:

   - **External metrics** (using labels):
     - NMI: Normalized Mutual Information  
     - AMI: Adjusted Mutual Information  
     - ARI: Adjusted Rand Index  
     - FMI: Fowlkes–Mallows Index  
     - ACC: Clustering accuracy with optimal label permutation (Hungarian)
   - **Internal metrics** (geometry-only on predicted clusters):
     - Silhouette score  
     - Davies–Bouldin index  
     - Calinski–Harabasz index  

   And we generate a collection of diagnostic plots (all saved as transparent PNG under `outputs/<run_id>/figures/`):

   - **Train/Test scatter plots**
     - Train ground-truth clusters
     - Train predicted clusters (title only shows NMI + ACC to avoid clutter)
     - Test points with correct vs mis-clustered points highlighted (again NMI + ACC)
   - **Full-dataset ground truth scatter** (entire \(X\), not just train)
   - **Micro-cluster graph visualization**
     - Micro-cluster centroids, colored by majority label
     - Edges drawn between connected micro-clusters
   - **Micro-cluster feature histograms**
     - `log(1 + size)`, `radius_mean`, `radius_std`, `log_density`
   - **Graph degree histogram** (node degree distribution)
   - **Edge feature histograms**
     - `center_dist`, `border_min_dist`, `log_density_diff`
     - `log_density_i`, `log_density_j`
   - **Distance matrix heatmap**
     - Pairwise micro-centroid distance matrix
   - **Pair-Transformers training curves**
     - BCE loss vs epoch
     - Pairwise accuracy vs epoch
   - **Pairwise probability histograms**
     - All \(P(	ext{merge})\)
     - \(P(	ext{merge})\) split for “same majority label” vs “different majority label”
   - **Per-metric bar plots (train vs test)**
     - One figure per metric: NMI, AMI, ARI, FMI, Silhouette, Davies–Bouldin, Calinski–Harabasz, ACC.

---

## Project structure

```text
gothic_project/
  helper.py          # utilities: set_seed, majority_label, accuracy with mapping, UnionFind
  dataset.py         # download/generate datasets into datasets/synthetic and datasets/real
  loader.py          # load datasets by name (X, y)
  gothic_model.py    # main implementation of GOTHIC (CLI-capable)
  run_gothic.py      # convenience wrapper for GOTHIC only
  runner.py          # generic runner: select model via --model (e.g., gothic)
  README.md          # this file

  datasets/          # (created when you download data)
    synthetic/       # SIPU-style txt datasets, noisy_circles.csv
    real/            # breast_cancer.csv, iris.csv, wine.csv

  outputs/           # (created when you run models)
    <run_id>/        # run-specific outputs (e.g., ID001)
      checkpoints/   # saved models
      figures/       # all PNG plots (transparent)
        *_micro_graph.png
        *_train_test.png
        *_full_ground_truth.png
        *_micro_features_hist.png
        *_micro_graph_degree_hist.png
        *_edge_features_hist.png
        *_edge_density_features_hist.png
        *_micro_distance_matrix.png
        *_pair_training_curves.png
        *_pair_score_hist.png
        *_metric_*.png      # one per metric (NMI/AMI/ARI/FMI/Sil/DB/CH/ACC)
      *_results.txt  # metrics summary for the run
```

---

## Installation

You’ll need:

- Python 3.9+ (3.11 recommended)
- Packages listed in `requirements.txt`

Basic install:

```bash
pip install -r requirements.txt
```

---

## Datasets

### Supported synthetic datasets (`datasets/synthetic/`)

These come from the SIPU dataset collection and are downloaded as `.txt`:

- `compound` → `Compound.txt`
- `aggregation` → `Aggregation.txt`
- `d31` → `D31.txt`
- `flame` → `flame.txt`
- `jain` → `jain.txt`
- `pathbased` → `pathbased.txt`
- `r15` → `R15.txt`

And an extra synthetic dataset:

- `noisy_circles` → **generated** via `sklearn.make_circles` and saved as `noisy_circles.csv` with columns:
  - `x1, x2, label`

### Supported real datasets (`datasets/real/`)

These are exported from scikit-learn and stored as CSV:

- `breast_cancer` → `breast_cancer.csv`
- `iris` → `iris.csv`
- `wine` → `wine.csv`

All real CSVs have:
- feature columns first, and
- a final `label` column.

---

## `dataset.py` – downloading / generating datasets

`dataset.py` ensures the datasets exist on disk.

### CLI

```bash
python dataset.py                       # download/generate ALL supported datasets
python dataset.py --datasets compound   # only Compound
python dataset.py --datasets compound aggregation noisy_circles iris
```

#### Arguments

- `--datasets` (optional, one or more names)  
  - If omitted or empty → **all** supported datasets are downloaded/generated.
  - Supported names:
    - Synthetic: `compound`, `aggregation`, `d31`, `flame`, `jain`, `pathbased`, `r15`, `noisy_circles`
    - Real: `breast_cancer`, `iris`, `wine`

### Programmatic usage

```python
from dataset import ensure_dataset_exists

path = ensure_dataset_exists("compound")      # ensures datasets/synthetic/Compound.txt
path = ensure_dataset_exists("noisy_circles") # ensures datasets/synthetic/noisy_circles.csv
path = ensure_dataset_exists("iris")          # ensures datasets/real/iris.csv
```

---

## `loader.py` – loading datasets

`loader.py` loads a dataset by name and returns `(X, y)` as NumPy arrays.

### Usage

```python
from loader import load_dataset

X, y = load_dataset("compound")       # X: (N, 2),   y: (N,)
X, y = load_dataset("noisy_circles")  # X: (N, 2),   y: (N,)
X, y = load_dataset("iris")           # X: (N, d),   y: (N,)
```

Behavior:

- Synthetic SIPU `.txt` datasets are assumed to have **3 columns**: `x, y, label (1..K)`.  
  Labels are shifted to `0..K-1`.
- `noisy_circles.csv` has 3 columns: `x1, x2, label (0..1)`.
- Real CSV datasets (breast_cancer, iris, wine) are loaded with all but last column as features, and last column as integer labels.

If the file does not exist, `load_dataset` calls `dataset.ensure_dataset_exists` to download/generate it.

---

## `gothic_model.py` – the GOTHIC implementation

This is the main script implementing the GOTHIC pipeline, including:

- over-clustering,
- graph construction,
- Transformer-based pair scoring,
- hierarchical merging,
- and a rich evaluation + plotting suite.

### Default dataset-specific `k_target`

If you do **not** specify `--k_target` or you pass a value ≤ 0, GOTHIC uses these defaults:

```python
DEFAULT_K_TARGETS = {
    "compound":      6,
    "aggregation":   7,
    "d31":           31,
    "flame":         2,
    "jain":          2,
    "pathbased":     3,
    "r15":           15,
    "noisy_circles": 2,
    "breast_cancer": 2,
    "iris":          3,
    "wine":          3,
}
```

### CLI interface

You can call it directly:

```bash
python gothic_model.py --dataset compound --k_target 6
```

or via `runner.py` / `run_gothic.py` (see below).

#### All arguments

```text
--dataset <str>            Dataset name (synthetic or real). Default: compound.
                           Supported: compound, aggregation, d31, flame, jain,
                           pathbased, r15, noisy_circles, breast_cancer, iris, wine.

--seed <int>               Random seed (Python, NumPy, PyTorch). Default: 42.

--train_fraction <float>   Fraction of data used for training (0-1). Default: 0.8.

--split_strategy <str>     Train/test split strategy. Default: balanced.
                           Options:
                             - random   : random 80/20 split
                             - balanced : density-aware per-cluster split
                                          (keeps more boundary points in train).

--data_root <str>          Base directory for datasets. Default: datasets.

--out_dir <str>            Base directory for outputs. Default: outputs.

--run_id <str>             Run identifier. All outputs go under:
                             out_dir / run_id /
                           Default: "default".

--n_micro <int>            Number of KMeans micro-clusters for over-clustering. Default: 80.

--kmeans_n_init <int>      KMeans n_init parameter. Default: 50.

--kmeans_max_iter <int>    KMeans max_iter. Default: 500.

--dist_quantile <float>    Quantile for thresholding centroid distances to form edges
                           in the micro-cluster graph. Default: 0.3.
                           Smaller -> sparser graph; larger -> denser graph.

--b_boundary <int>         Number of boundary points per micro-cluster used for
                           border-distance computation. Default: 5.

--d_model <int>            Transformer hidden dimension (node embedding size).
                           Default: 32.

--n_heads <int>            Number of attention heads in Transformer. Default: 4.

--attn_hidden <int>        Hidden dimension in the pairwise MLP. Default: 64.

--n_transformer_layers <int>
                           Number of Transformer encoder layers. Default: 2.

--train_epochs <int>       Number of epochs for training the pairwise Transformer.
                           Default: 2000.

--lr <float>               Learning rate for Adam optimizer. Default: 1e-3.

--weight_decay <float>     Weight decay (L2 regularization) in Adam. Default: 1e-4.

--print_every <int>        How frequently to print training stats. Default: 200.

--k_target <int>           Target number of macro-clusters. If <= 0, use dataset-
                           specific default (see table above).

--show_plots               If provided, show matplotlib windows interactively
                           (in addition to saving PNG files).
                           This is a boolean flag; no value required.
```

### Outputs and directory layout

For a command like:

```bash
python gothic_model.py --dataset compound --k_target 6 --run_id ID001 --out_dir outputs
```

you get:

```text
outputs/
  ID001/
    checkpoints/
      gothic_pair_transformer_best_compound.pt   # best pairwise model
    figures/
      compound_micro_graph.png
      compound_train_test.png
      compound_full_ground_truth.png
      compound_micro_features_hist.png
      compound_micro_graph_degree_hist.png
      compound_edge_features_hist.png
      compound_edge_density_features_hist.png
      compound_micro_distance_matrix.png
      compound_pair_training_curves.png
      compound_pair_score_hist.png
      compound_metric_nmi.png
      compound_metric_ami.png
      compound_metric_ari.png
      compound_metric_fmi.png
      compound_metric_silhouette.png
      compound_metric_daviesbouldin.png
      compound_metric_calinskiharabasz.png
      compound_metric_acc.png
    compound_results.txt                         # metrics summary (train/test)
```

`*_results.txt` contains:

- dataset, run_id
- seed, split_strategy, train_fraction
- n_micro, k_target, dist_quantile
- All **train** + **test** metrics:
  - NMI, AMI, ARI, FMI, Silhouette, Davies–Bouldin, Calinski–Harabasz, ACC

---

## `runner.py` – generic model runner with `--model`

`runner.py` is a generic entry point that chooses which model to run based on `--model`.

Currently supported:

- `--model gothic` → runs `gothic_model.py`.

Everything **after** `--model ...` is passed unchanged to the chosen model script.

### Usage

```bash
# Run GOTHIC on Compound with full hyperparameters
python runner.py --model gothic --run_id ID001 --dataset compound --seed 42 --train_fraction 0.8 --split_strategy balanced --n_micro 80 --kmeans_n_init 50 --kmeans_max_iter 500 --dist_quantile 0.3 --b_boundary 5 --d_model 32 --n_heads 4 --attn_hidden 64 --n_transformer_layers 2 --train_epochs 2000 --lr 0.001 --weight_decay 0.0001 --k_target 6 --data_root datasets --out_dir outputs --show_plots
```

The above is equivalent to calling `gothic_model.py` directly with the same arguments.

Later, if you add other models (e.g., `--model spectral`, `--model dbscan_transformer`), you simply:

- create a corresponding `*_model.py` file, and
- add a branch in `runner.py` to map `--model` to that script.

---

## Quick start examples

### 1. Download all datasets

```bash
python dataset.py
```

### 2. Run GOTHIC on Compound (minimal)

Uses defaults: `seed=42`, `train_fraction=0.8`, `split_strategy=balanced`,
`n_micro=80`, `k_target=6` (from dataset-specific defaults), etc.

```bash
python runner.py --model gothic --dataset compound --run_id cmp_default
```

### 3. Run GOTHIC on Compound with explicit hyperparams

```bash
python runner.py --model gothic --run_id ID001 --dataset compound --seed 42 --train_fraction 0.8 --split_strategy balanced --n_micro 80 --kmeans_n_init 50 --kmeans_max_iter 500 --dist_quantile 0.3 --b_boundary 5 --d_model 32 --n_heads 4 --attn_hidden 64 --n_transformer_layers 2 --train_epochs 2000 --lr 0.001 --weight_decay 0.0001 --k_target 6 --data_root datasets --out_dir outputs --show_plots
```

### 4. Run GOTHIC on Iris

```bash
python runner.py --model gothic --run_id IRIS01 --dataset iris --k_target 3 --n_micro 40
```

### 5. Run GOTHIC on noisy_circles

```bash
python runner.py --model gothic --run_id CIRC01 --dataset noisy_circles --k_target 2 --n_micro 50
```

---

## Reproducibility notes

- `--seed` controls random seeds for:
  - Python’s `random`,
  - NumPy,
  - PyTorch (CPU and CUDA, if available).
- `--run_id` is intended as an experiment ID — you can run:
  - `run_id=cmp01`, `cmp02`, `cmp03` … and inspect each directory separately.
- For exact reproducibility, keep:
  - dataset,
  - all CLI arguments,
  - library versions (NumPy, scikit-learn, PyTorch, …).

---

## Extending GOTHIC & the project

- You can modify GOTHIC’s internals (e.g., add more node features, pair features, or different merging strategies) **without** changing the CLI.
- To add **another model** alongside GOTHIC:
  1. Create a new script, e.g. `my_model.py` with its own `main(args)`.
  2. Update `runner.py` to map `--model my_model` to that script.
  3. Use `runner.py --model my_model ...` to run it.

This design lets you build a small library of clustering models all sharing:

- the same dataset helpers (`dataset.py`, `loader.py`),
- the same experiment management (`run_id`, `out_dir`),
- and a uniform CLI pattern.
