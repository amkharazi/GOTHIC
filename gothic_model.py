"""
gothic_model.py

Core implementation of GOTHIC:
  Graph-Overclustered Transformer-based Hierarchical Integrated Clustering.

Pipeline (unchanged in spirit):
  1) Load dataset (X,y) via loader.load_dataset.
  2) Train/test split (random or density-balanced).
  3) (NEW, safe) Optional automatic PCA fallback for high-d.
  4) Over-cluster train set via KMeans into micro-clusters.
  5) Build micro-cluster graph (distance quantile) + pairwise features.
  6) Train Transformer-based pair scorer on the micro-cluster graph (supervised).
  7) Greedy merging via learned scores until K_TARGET (fallback: distance merges).
  8) Label train & test, compute metrics (NMI, ACC, AMI, ARI, FMI, Sil, DB, CH).
  9) Save plots, metrics, and checkpoint to disk under outputs/<run_id>/.
  10) Append summary row to results_summary.csv.

Fixes included (without changing the method):
  - Overflow fix in micro "density" feature for very high dimension:
      replaces (radius_mean ** d) with a stable log-space form + capped d_eff
  - Optional auto PCA fallback for high dimensional data (e.g., olivetti_faces d=4096)
      so distances/KMeans/graph construction are meaningful and stable.
  - Plotting uses PCA-to-2D when dims > 2 (instead of blindly first 2 features).

Run:
  python gothic_model.py --dataset wine --run_id wine_ID001 --k_target 3
"""

from __future__ import annotations

MODEL_NAME = "gothic"

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    adjusted_rand_score,
    fowlkes_mallows_score,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    pairwise_distances,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure local imports work when run as a script
sys.path.append(str(Path(__file__).resolve().parent))

from helper import (
    set_seed,
    majority_label,
    clustering_accuracy_with_map,
    apply_label_mapping,
    UnionFind,
)
from loader import load_dataset


# ======================== DEFAULT CONFIG ===========================

DEFAULT_SEED = 42
DEFAULT_TRAIN_FRACTION = 0.8
DEFAULT_SPLIT_STRATEGY = "balanced"   # "random" or "balanced"
DEFAULT_DATASET = "compound"

DEFAULT_N_MICRO = 80              # Over-clustering K
DEFAULT_KMEANS_N_INIT = 50
DEFAULT_KMEANS_MAX_ITER = 500

DEFAULT_DIST_QUANTILE = 0.3       # edges between centroids whose distance <= this quantile
DEFAULT_B_BOUNDARY = 5            # number of boundary points per micro-cluster for border distances

DEFAULT_D_MODEL = 32
DEFAULT_N_HEADS = 4
DEFAULT_ATTN_HIDDEN = 64
DEFAULT_N_TRANSFORMER_LAYERS = 2
DEFAULT_TRAIN_EPOCHS = 2000
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_PRINT_EVERY = 200

# NEW: high-d stabilization defaults
DEFAULT_REDUCE_DIM = "none"          # "none" | "auto" | "pca"
DEFAULT_PCA_TRIGGER_DIM = 100        # apply PCA if d > this (only for auto/pca)
DEFAULT_PCA_DIM = 64                 # reduced dimension
DEFAULT_MAX_DENSITY_DIM = 64         # cap d used in micro density term (log-space)

# Dataset-specific default K_TARGET
DEFAULT_K_TARGETS = {
    "compound": 6,
    "aggregation": 7,
    "d31": 31,
    "flame": 2,
    "jain": 2,
    "pathbased": 3,
    "r15": 15,
    "noisy_circles": 2,
    "breast_cancer": 2,
    "iris": 3,
    "wine": 3,
    "digits": 10,
    "olivetti_faces": 40,
}


# =========================== ARG PARSER ============================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="GOTHIC clustering: graph + transformer-based micro-cluster merging.")

    # Core
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET,
                        help="Dataset name (synthetic or real) as defined in loader.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train_fraction", type=float, default=DEFAULT_TRAIN_FRACTION)
    parser.add_argument(
        "--split_strategy",
        type=str,
        default=DEFAULT_SPLIT_STRATEGY,
        choices=["random", "balanced"],
        help="Train/test split strategy."
    )
    parser.add_argument("--data_root", type=str, default="datasets",
                        help="Base directory where datasets are stored.")
    parser.add_argument("--out_dir", type=str, default="outputs",
                        help="Base directory to save checkpoints, plots, and results.")
    parser.add_argument("--run_id", type=str, default="default",
                        help="Run identifier; all outputs go under out_dir/run_id/")

    # Micro-clustering + graph
    parser.add_argument("--n_micro", type=int, default=DEFAULT_N_MICRO)
    parser.add_argument("--kmeans_n_init", type=int, default=DEFAULT_KMEANS_N_INIT)
    parser.add_argument("--kmeans_max_iter", type=int, default=DEFAULT_KMEANS_MAX_ITER)
    parser.add_argument("--dist_quantile", type=float, default=DEFAULT_DIST_QUANTILE)
    parser.add_argument("--b_boundary", type=int, default=DEFAULT_B_BOUNDARY)

    # Transformer
    parser.add_argument("--d_model", type=int, default=DEFAULT_D_MODEL)
    parser.add_argument("--n_heads", type=int, default=DEFAULT_N_HEADS)
    parser.add_argument("--attn_hidden", type=int, default=DEFAULT_ATTN_HIDDEN)
    parser.add_argument("--n_transformer_layers", type=int, default=DEFAULT_N_TRANSFORMER_LAYERS)
    parser.add_argument("--train_epochs", type=int, default=DEFAULT_TRAIN_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--print_every", type=int, default=DEFAULT_PRINT_EVERY)

    # Target clusters
    parser.add_argument(
        "--k_target",
        type=int,
        default=-1,
        help="Target number of clusters; if <= 0, uses dataset-specific default."
    )

    # NEW: high-d stabilization
    parser.add_argument("--reduce_dim", type=str, default=DEFAULT_REDUCE_DIM, choices=["none", "auto", "pca"],
                        help="Dimensionality reduction mode. 'auto' applies PCA if d > pca_trigger_dim.")
    parser.add_argument("--pca_trigger_dim", type=int, default=DEFAULT_PCA_TRIGGER_DIM,
                        help="If reduce_dim is auto/pca: apply PCA when original d exceeds this value.")
    parser.add_argument("--pca_dim", type=int, default=DEFAULT_PCA_DIM,
                        help="If PCA is applied, reduce to this many components (capped by n_train-1 and d).")
    parser.add_argument("--max_density_dim", type=int, default=DEFAULT_MAX_DENSITY_DIM,
                        help="Cap on 'd' used in micro density proxy to prevent extreme-d domination/instability.")

    # Misc
    parser.add_argument("--show_plots", action="store_true",
                        help="If set, show matplotlib figures interactively after saving.")

    return parser.parse_args(argv)


# =========================== CSV HELPERS ============================

def _to_csv_value(v):
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, Path):
        return str(v)
    return v


def append_row_to_csv(csv_path: Path, row: Dict[str, Any]) -> None:
    """
    Append a row to csv_path. If file doesn't exist, create it with header.
    If file exists but header is missing new columns, rewrite with union header.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    row = {k: _to_csv_value(v) for k, v in row.items()}

    if not csv_path.exists():
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        return

    # Existing header
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            header = []

    if not header:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        return

    header_set = set(header)
    missing = [k for k in row.keys() if k not in header_set]

    if missing:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            old_reader = csv.DictReader(f)
            old_rows = list(old_reader)
            old_fieldnames = old_reader.fieldnames or header

        new_fieldnames = list(old_fieldnames) + missing

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=new_fieldnames)
            writer.writeheader()
            for r in old_rows:
                writer.writerow({k: r.get(k, "") for k in new_fieldnames})
            writer.writerow({k: row.get(k, "") for k in new_fieldnames})
        return

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writerow({k: row.get(k, "") for k in header})


# =========================== SPLIT HELPERS ==========================

def train_test_split_with_strategy(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float = 0.8,
    strategy: str = "balanced",
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Same behavior as before:
      - 'random'   : random split.
      - 'balanced' : per-class, density-aware split (densest interior points to test).
    """
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    indices = np.arange(n)

    if strategy == "random":
        perm = rng.permutation(indices)
        n_train = int(round(train_frac * n))
        train_idx = perm[:n_train]
        test_idx = perm[n_train:]

    elif strategy == "balanced":
        train_idx: List[int] = []
        test_idx: List[int] = []
        eps = 1e-8

        for c in np.unique(y):
            idx_c = indices[y == c]
            X_c = X[idx_c]
            n_c = len(idx_c)
            if n_c <= 1:
                train_idx.extend(idx_c)
                continue

            n_train_c = max(1, int(round(train_frac * n_c)))
            n_test_c = n_c - n_train_c
            if n_test_c <= 0:
                train_idx.extend(idx_c)
                continue

            k_nn = min(10, n_c - 1)
            nn = NearestNeighbors(n_neighbors=k_nn + 1)
            nn.fit(X_c)
            dists, _ = nn.kneighbors(X_c)
            mean_dists = dists[:, 1:].mean(axis=1)
            densities = 1.0 / (eps + mean_dists)

            order = np.argsort(-densities)
            test_local = order[:n_test_c]
            train_local = order[n_test_c:]

            train_idx.extend(idx_c[train_local])
            test_idx.extend(idx_c[test_local])

        train_idx = np.array(train_idx)
        test_idx = np.array(test_idx)

    else:
        raise ValueError(f"Unknown split strategy: {strategy}")

    return (
        X[train_idx], X[test_idx],
        y[train_idx], y[test_idx],
        train_idx, test_idx
    )


# =========================== SAFE METRICS ===========================

def safe_cluster_metric(fn, X_data, labels_pred, name: str) -> float:
    try:
        if len(labels_pred) == 0 or len(np.unique(labels_pred)) < 2:
            return float("nan")
        return float(fn(X_data, labels_pred))
    except Exception as e:
        print(f"[WARN] Failed to compute {name}: {e}")
        return float("nan")


def to_2d_for_plot(X: np.ndarray, seed: int = 42) -> np.ndarray:
    if X.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if X.shape[1] >= 2:
        # If > 2 dims, better to PCA for visualization.
        if X.shape[1] > 2:
            p = PCA(n_components=2, random_state=seed)
            return p.fit_transform(X).astype(np.float32)
        return X[:, :2].astype(np.float32)
    # 1D
    return np.column_stack([X[:, 0], np.zeros_like(X[:, 0])]).astype(np.float32)


# ======================= HIGH-D PREPROCESS ==========================

class FeaturePreprocessor:
    """
    Fit-on-train, apply-to-test preprocessor:
      - optional StandardScaler
      - optional PCA (auto/pca modes)
    """
    def __init__(self, reduce_dim: str, pca_trigger_dim: int, pca_dim: int, seed: int):
        self.reduce_dim = reduce_dim
        self.pca_trigger_dim = int(pca_trigger_dim)
        self.pca_dim = int(pca_dim)
        self.seed = int(seed)

        self._scaler: Optional[StandardScaler] = None
        self._pca: Optional[PCA] = None

        self.did_pca = False
        self.out_dim: Optional[int] = None

    def fit_transform_train(self, X_train: np.ndarray) -> np.ndarray:
        X = X_train

        d = X.shape[1]
        do_pca = False
        if self.reduce_dim == "pca":
            do_pca = True
        elif self.reduce_dim == "auto":
            do_pca = d > self.pca_trigger_dim

        if do_pca:
            n_train = X.shape[0]
            ncomp = min(self.pca_dim, max(1, n_train - 1), d)
            self._pca = PCA(
                n_components=ncomp,
                random_state=self.seed,
                svd_solver="randomized" if d > 200 else "auto",
            )
            X = self._pca.fit_transform(X)
            self.did_pca = True

        self.out_dim = X.shape[1]
        return X.astype(np.float32)

    def transform_test(self, X_test: np.ndarray) -> np.ndarray:
        X = X_test
        if self._scaler is not None:
            X = self._scaler.transform(X)
        if self._pca is not None:
            X = self._pca.transform(X)
        return X.astype(np.float32)


# ======================= MICRO-CLUSTER BUILD ========================

def stable_log_density(size: int, radius_mean: float, d: int, max_d: int, eps: float = 1e-12) -> float:
    """
    Stable replacement for:
        log(1 + size / (radius_mean**d + eps))

    We compute:
        x = log(size) - d_eff*log(radius_mean)
        log(1 + exp(x))  (softplus)  -> stable

    Also cap d_eff to avoid extreme-d domination and to prevent numerical issues.
    """
    d_eff = int(min(max(1, d), max(1, max_d)))
    r = max(float(radius_mean), eps)
    s = max(float(size), eps)

    x = math.log(s) - d_eff * math.log(r)  # log(size / r^d_eff)

    # softplus(x) = log(1 + exp(x)) stable
    if x > 30:
        return float(x)
    if x < -30:
        return float(math.exp(x))
    return float(math.log1p(math.exp(x)))


def build_micro_clusters(
    X_train_use: np.ndarray,
    y_train: np.ndarray,
    n_micro: int,
    seed: int,
    kmeans_n_init: int,
    kmeans_max_iter: int,
    b_boundary: int,
    max_density_dim: int,
) -> Dict[str, Any]:
    """
    Over-cluster with KMeans on train; build micro features.
    Node feature = [centroid (d_use dims), log(1+size), radius_mean, radius_std, log_density]
    """
    print(f"[INFO] Over-clustering train set into {n_micro} micro-clusters ...")
    km = KMeans(
        n_clusters=n_micro,
        random_state=seed,
        n_init=kmeans_n_init,
        max_iter=kmeans_max_iter,
    )
    km.fit(X_train_use)
    train_micro_orig = km.labels_

    micro_features: List[np.ndarray] = []
    micro_major_labels: List[int] = []
    orig_to_new: Dict[int, int] = {}
    new_to_orig: List[int] = []
    micro_member_indices: List[np.ndarray] = []
    micro_border_indices: List[np.ndarray] = []

    d_use = X_train_use.shape[1]

    for m in range(n_micro):
        idx_m = np.where(train_micro_orig == m)[0]
        if len(idx_m) == 0:
            continue

        pts = X_train_use[idx_m]
        centroid = pts.mean(axis=0)
        radii = np.linalg.norm(pts - centroid, axis=1)
        radius_mean = float(radii.mean())
        radius_std = float(radii.std()) if len(radii) > 1 else 0.0
        size = int(len(pts))
        mj = int(majority_label(y_train[idx_m]))

        # FIX: stable log density (no overflow)
        log_density = stable_log_density(size=size, radius_mean=radius_mean, d=d_use, max_d=max_density_dim)

        scalar_feat = np.array(
            [math.log(1.0 + size), radius_mean, radius_std, log_density],
            dtype=np.float32,
        )
        feat = np.concatenate([centroid.astype(np.float32), scalar_feat], axis=0)

        # Boundary points: farthest from centroid
        b_eff = min(max(1, int(b_boundary)), len(idx_m))
        order_r = np.argsort(-radii)
        boundary_local = order_r[:b_eff]
        boundary_global_idx = idx_m[boundary_local]

        new_id = len(micro_features)
        orig_to_new[m] = new_id
        new_to_orig.append(m)
        micro_features.append(feat)
        micro_major_labels.append(mj)
        micro_member_indices.append(idx_m)
        micro_border_indices.append(boundary_global_idx)

    micro_features_arr = np.stack(micro_features, axis=0) if micro_features else np.zeros((0, d_use + 4), dtype=np.float32)
    micro_major_labels_arr = np.array(micro_major_labels, dtype=int) if micro_major_labels else np.zeros((0,), dtype=int)

    print(f"[INFO] Effective non-empty micro-clusters: {len(micro_features_arr)}")
    return {
        "kmeans": km,
        "train_micro_orig": train_micro_orig,
        "micro_features": micro_features_arr,
        "micro_major_labels": micro_major_labels_arr,
        "orig_to_new": orig_to_new,
        "new_to_orig": new_to_orig,
        "micro_member_indices": micro_member_indices,
        "micro_border_indices": micro_border_indices,
    }


# ======================= GRAPH + PAIR FEATURES ======================

def build_edges_with_pair_features(
    micro_features: np.ndarray,
    micro_border_indices: List[np.ndarray],
    X_train_use: np.ndarray,
    dist_quantile: float = 0.3,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Build edges between micro-cluster centroids using a distance threshold,
    compute pairwise features:
      - center_dist
      - border_min_dist (between boundary points)
      - log_density_diff
      - log_density_i
      - log_density_j

    micro_features: [K_eff, d_use+4], last 4 dims are [log_size, radius_mean, radius_std, log_density].
    """
    if micro_features.shape[0] == 0:
        return np.zeros((0, 2), dtype=int), float("nan"), np.zeros((0, 5), dtype=np.float32)

    d_use = X_train_use.shape[1]
    centroids = micro_features[:, :d_use]
    log_density = micro_features[:, d_use + 3]

    D = pairwise_distances(centroids, metric="euclidean")
    K_eff = centroids.shape[0]
    iu = np.triu_indices(K_eff, k=1)
    dvals = D[iu]
    thresh = float(np.quantile(dvals, dist_quantile)) if len(dvals) else 0.0

    edge_list: List[Tuple[int, int]] = []
    pair_extra: List[Tuple[float, float, float, float, float]] = []

    for i, j in zip(iu[0], iu[1]):
        center_dist = float(D[i, j])
        if center_dist > thresh:
            continue

        idx_i = micro_border_indices[i]
        idx_j = micro_border_indices[j]
        pts_i = X_train_use[idx_i] if len(idx_i) else np.zeros((0, d_use), dtype=np.float32)
        pts_j = X_train_use[idx_j] if len(idx_j) else np.zeros((0, d_use), dtype=np.float32)

        if len(pts_i) == 0 or len(pts_j) == 0:
            border_min_dist = center_dist
        else:
            Dij = pairwise_distances(pts_i, pts_j, metric="euclidean")
            border_min_dist = float(Dij.min())

        ld_i = float(log_density[i])
        ld_j = float(log_density[j])
        ld_diff = float(abs(ld_i - ld_j))

        edge_list.append((i, j))
        pair_extra.append((center_dist, border_min_dist, ld_diff, ld_i, ld_j))

    edge_arr = np.array(edge_list, dtype=int) if edge_list else np.zeros((0, 2), dtype=int)
    extra_arr = np.array(pair_extra, dtype=np.float32) if pair_extra else np.zeros((0, 5), dtype=np.float32)

    print(f"[INFO] Built {len(edge_arr)} edges with dist <= {thresh:.4f} (quantile={dist_quantile})")
    return edge_arr, thresh, extra_arr


def plot_micro_graph(micro_features: np.ndarray, micro_major_labels: np.ndarray, edge_list: np.ndarray, dist_thresh: float,
                     seed: int) -> plt.Figure:
    if micro_features.shape[0] == 0:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.set_title("Micro-cluster graph (empty)")
        ax.axis("off")
        fig.tight_layout()
        return fig

    # plot in 2D (PCA if needed)
    d_use = micro_features.shape[1] - 4
    centroids = micro_features[:, :d_use]
    cent2 = to_2d_for_plot(centroids, seed=seed)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(
        cent2[:, 0],
        cent2[:, 1],
        c=micro_major_labels,
        s=60,
        cmap="tab10",
        edgecolors="black",
        alpha=0.9,
    )

    for (i, j) in edge_list:
        x1, y1 = cent2[i, 0], cent2[i, 1]
        x2, y2 = cent2[j, 0], cent2[j, 1]
        ax.plot([x1, x2], [y1, y2], linestyle="-", linewidth=0.5, alpha=0.5, color="gray")

    ax.set_title(f"Micro-cluster graph (dist <= {dist_thresh:.3f})")
    ax.set_xlabel("dim-1")
    ax.set_ylabel("dim-2")
    fig.tight_layout()
    return fig


# =================== TRANSFORMER-BASED PAIR SCORER ==================

class PairTransformerNet(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 32, n_heads: int = 4, hidden_dim: int = 64,
                 n_layers: int = 2, pair_extra_dim: int = 0):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        in_dim = 4 * d_model + int(pair_extra_dim)
        self.pair_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_features: torch.Tensor, pair_idx: torch.Tensor, pair_extra: Optional[torch.Tensor] = None):
        """
        node_features: [K, input_dim]
        pair_idx:      [M, 2] int64
        pair_extra:    [M, E] float32 or None
        """
        h0 = self.proj(node_features)          # [K, d_model]
        h_enc = self.encoder(h0.unsqueeze(0))  # [1, K, d_model]
        h = h_enc.squeeze(0)                   # [K, d_model]

        hi = h[pair_idx[:, 0]]
        hj = h[pair_idx[:, 1]]

        z = torch.cat([hi, hj, torch.abs(hi - hj), hi * hj], dim=1)
        if pair_extra is not None:
            z = torch.cat([z, pair_extra], dim=1)

        logits = self.pair_mlp(z).squeeze(1)
        return logits


def train_pair_transformer(
    micro_features: np.ndarray,
    micro_major_labels: np.ndarray,
    edge_list: np.ndarray,
    pair_extra: np.ndarray,
    device: torch.device,
    args,
    dataset: str,
    out_dir: Path,
    figures_dir: Path,
) -> PairTransformerNet:
    X_nodes = torch.tensor(micro_features, dtype=torch.float32, device=device)

    if len(edge_list) == 0:
        raise RuntimeError("Edge list is empty; cannot train pair model. Try increasing --dist_quantile or adjusting n_micro.")

    pair_idx = torch.tensor(edge_list, dtype=torch.long, device=device)
    pair_extra_tensor = torch.tensor(pair_extra, dtype=torch.float32, device=device) if pair_extra.size else None

    labels_np = (micro_major_labels[edge_list[:, 0]] == micro_major_labels[edge_list[:, 1]]).astype(np.float32)
    y_pairs = torch.tensor(labels_np, dtype=torch.float32, device=device)

    model = PairTransformerNet(
        input_dim=micro_features.shape[1],
        d_model=args.d_model,
        n_heads=args.n_heads,
        hidden_dim=args.attn_hidden,
        n_layers=args.n_transformer_layers,
        pair_extra_dim=pair_extra.shape[1] if pair_extra.size else 0,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_acc = -1.0
    best_loss = float("inf")
    best_state = None

    loss_history: List[float] = []
    acc_history: List[float] = []

    model.train()
    for epoch in range(1, args.train_epochs + 1):
        optimizer.zero_grad()
        logits = model(X_nodes, pair_idx, pair_extra_tensor)
        loss = criterion(logits, y_pairs)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            acc = (preds == y_pairs).float().mean().item()
            loss_val = float(loss.item())

        loss_history.append(loss_val)
        acc_history.append(acc)

        if (acc > best_acc) or (math.isclose(acc, best_acc) and loss_val < best_loss):
            best_acc = acc
            best_loss = loss_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch % args.print_every == 0 or epoch == 1 or epoch == args.train_epochs:
            print(
                f"[TRAIN] Epoch {epoch:4d}/{args.train_epochs}, "
                f"loss={loss_val:.4f}, pair-acc={acc:.4f}, "
                f"best_acc={best_acc:.4f}, best_loss={best_loss:.4f}"
            )

    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"gothic_pair_transformer_best_{dataset}.pt"

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, ckpt_path)
        print(f"[INFO] Saved best PairTransformerNet weights to: {ckpt_path}")
    else:
        torch.save(model.state_dict(), ckpt_path)
        print(f"[WARN] No best_state recorded; saved last-epoch model instead at: {ckpt_path}")

    # Curves
    try:
        epochs = np.arange(1, len(loss_history) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(epochs, loss_history, linewidth=1.5)
        axes[0].set_title("PairTransformer training loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("BCE loss")

        axes[1].plot(epochs, acc_history, linewidth=1.5)
        axes[1].set_title("PairTransformer training accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Pair accuracy")

        fig.tight_layout()
        path = figures_dir / f"{dataset}_pair_training_curves.png"
        fig.savefig(path, dpi=150, transparent=True)
        print(f"[INFO] Saved training curves figure to: {path}")
        if args.show_plots:
            fig.show()
        else:
            plt.close(fig)
    except Exception as e:
        print(f"[WARN] Failed to save training curves plot: {e}")

    # Pair score histogram
    try:
        model.eval()
        with torch.no_grad():
            logits_best = model(X_nodes, pair_idx, pair_extra_tensor)
            probs_best = torch.sigmoid(logits_best).cpu().numpy()
        y_np = y_pairs.cpu().numpy()

        pos = probs_best[y_np == 1]
        neg = probs_best[y_np == 0]

        fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
        axes2[0].hist(probs_best, bins=30)
        axes2[0].set_title("All pairwise merge probabilities")
        axes2[0].set_xlabel("P(merge)")
        axes2[0].set_ylabel("Count")
        axes2[0].set_xlim(0.0, 1.0)

        if len(pos) > 0:
            axes2[1].hist(pos, bins=30, alpha=0.6, label=f"Same label (n={len(pos)})")
        if len(neg) > 0:
            axes2[1].hist(neg, bins=30, alpha=0.6, label=f"Different label (n={len(neg)})")
        axes2[1].set_title("P(merge) by majority-label relation")
        axes2[1].set_xlabel("P(merge)")
        axes2[1].set_ylabel("Count")
        axes2[1].set_xlim(0.0, 1.0)
        axes2[1].legend()

        fig2.tight_layout()
        path2 = figures_dir / f"{dataset}_pair_score_hist.png"
        fig2.savefig(path2, dpi=150, transparent=True)
        print(f"[INFO] Saved pair-score histogram to: {path2}")
        if args.show_plots:
            fig2.show()
        else:
            plt.close(fig2)
    except Exception as e:
        print(f"[WARN] Failed to save pair-score histogram: {e}")

    return model


def merge_micro_clusters_learned(
    micro_features: np.ndarray,
    edge_list: np.ndarray,
    pair_extra: np.ndarray,
    model: PairTransformerNet,
    device: torch.device,
    k_target: int,
) -> np.ndarray:
    K_eff = micro_features.shape[0]
    uf = UnionFind(K_eff)

    if K_eff == 0:
        return np.zeros((0,), dtype=int)

    X_nodes = torch.tensor(micro_features, dtype=torch.float32, device=device)
    pair_idx = torch.tensor(edge_list, dtype=torch.long, device=device)
    pair_extra_tensor = torch.tensor(pair_extra, dtype=torch.float32, device=device) if pair_extra.size else None

    model.eval()
    with torch.no_grad():
        logits = model(X_nodes, pair_idx, pair_extra_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()

    order = np.argsort(-probs)
    for idx in order:
        i, j = edge_list[idx]
        if uf.set_count <= k_target:
            break
        uf.union(int(i), int(j))

    print(f"[INFO] After learned merges, components = {uf.set_count}")

    if uf.set_count > k_target:
        print("[INFO] Falling back to distance-based merges to reach k_target.")
        d_use = micro_features.shape[1] - 4
        centroids = micro_features[:, :d_use]
        D = pairwise_distances(centroids, metric="euclidean")
        iu = np.triu_indices(K_eff, k=1)
        dist_pairs = list(zip(iu[0], iu[1], D[iu]))
        dist_pairs.sort(key=lambda t: float(t[2]))
        for i, j, _ in dist_pairs:
            if uf.set_count <= k_target:
                break
            uf.union(int(i), int(j))

    print(f"[INFO] Final number of macro-clusters: {uf.set_count}")
    roots = sorted({uf.find(i) for i in range(K_eff)})
    root_to_macro = {r: idx for idx, r in enumerate(roots)}

    micro_to_macro = np.zeros(K_eff, dtype=int)
    for i in range(K_eff):
        micro_to_macro[i] = root_to_macro[uf.find(i)]

    return micro_to_macro


# ============================= MAIN ================================

def gothic_main(args=None) -> Dict[str, Any]:
    if args is None:
        args = parse_args()

    dataset = args.dataset.lower()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Outputs will be saved under: {out_dir}")
    print(f"[INFO] Figures will be saved under: {figures_dir}")

    # k_target
    if args.k_target is None or args.k_target <= 0:
        k_target = DEFAULT_K_TARGETS.get(dataset, 6)
        print(f"[INFO] Using dataset-specific k_target={k_target} for dataset={dataset}")
    else:
        k_target = int(args.k_target)
        print(f"[INFO] Using user-specified k_target={k_target}")

    # Load dataset
    X, y = load_dataset(dataset, base_dir=str(data_root))
    X = X.astype(np.float32)
    y = y.astype(int)
    print(f"[INFO] Loaded dataset '{dataset}': X.shape={X.shape}, unique labels={np.unique(y)}")

    # Split (on original space)
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split_with_strategy(
        X, y, train_frac=args.train_fraction, strategy=args.split_strategy, seed=args.seed
    )
    print(f"[INFO] Train size: {len(X_train)}, Test size: {len(X_test)}")

    # NEW: high-d preprocessing (fit on train, apply to test)
    pre = FeaturePreprocessor(
        reduce_dim=str(args.reduce_dim),
        pca_trigger_dim=int(args.pca_trigger_dim),
        pca_dim=int(args.pca_dim),
        seed=int(args.seed),
    )
    X_train_use = pre.fit_transform_train(X_train)
    X_test_use = pre.transform_test(X_test)

    if pre.did_pca:
        print(f"[INFO] PCA enabled: d={X_train.shape[1]} -> d'={X_train_use.shape[1]}")

    # Step 3: Over-cluster on (possibly reduced) train
    mc = build_micro_clusters(
        X_train_use=X_train_use,
        y_train=y_train,
        n_micro=args.n_micro,
        seed=args.seed,
        kmeans_n_init=args.kmeans_n_init,
        kmeans_max_iter=args.kmeans_max_iter,
        b_boundary=args.b_boundary,
        max_density_dim=args.max_density_dim,
    )
    micro_features = mc["micro_features"]
    micro_major_labels = mc["micro_major_labels"]
    orig_to_new = mc["orig_to_new"]
    new_to_orig = mc["new_to_orig"]
    km = mc["kmeans"]
    train_micro_orig = mc["train_micro_orig"]
    micro_border_indices = mc["micro_border_indices"]

    if len(micro_features) == 0:
        raise RuntimeError("All micro-clusters empty (unexpected). Try smaller n_micro or verify dataset.")

    train_micro_new = np.array([orig_to_new[m] for m in train_micro_orig], dtype=int)

    # Micro scalar histograms
    try:
        d_use = X_train_use.shape[1]
        scalar_feats = micro_features[:, d_use:]
        feat_names = ["log(1+size)", "radius_mean", "radius_std", "log_density"]

        fig_micro, axes_micro = plt.subplots(2, 2, figsize=(10, 8))
        axes_micro = axes_micro.ravel()
        for i in range(4):
            axes_micro[i].hist(scalar_feats[:, i], bins=20)
            axes_micro[i].set_title(f"Micro feature: {feat_names[i]}")
            axes_micro[i].set_xlabel(feat_names[i])
            axes_micro[i].set_ylabel("Count")
        fig_micro.tight_layout()
        micro_hist_path = figures_dir / f"{dataset}_micro_features_hist.png"
        fig_micro.savefig(micro_hist_path, dpi=150, transparent=True)
        print(f"[INFO] Saved micro-feature histograms to: {micro_hist_path}")
        if args.show_plots:
            fig_micro.show()
        else:
            plt.close(fig_micro)
    except Exception as e:
        print(f"[WARN] Failed to save micro-feature histograms: {e}")

    # Step 4: Graph + pair features (in the same space as micro clustering)
    edge_list, dist_thresh, pair_extra = build_edges_with_pair_features(
        micro_features=micro_features,
        micro_border_indices=micro_border_indices,
        X_train_use=X_train_use,
        dist_quantile=args.dist_quantile,
    )

    # Degree histogram
    try:
        K_eff = micro_features.shape[0]
        degrees = np.zeros(K_eff, dtype=int)
        for (i, j) in edge_list:
            degrees[i] += 1
            degrees[j] += 1

        fig_deg, ax_deg = plt.subplots(figsize=(6, 4))
        if degrees.max() > 0:
            ax_deg.hist(degrees, bins=np.arange(degrees.max() + 2) - 0.5)
        else:
            ax_deg.hist(degrees, bins=3)
        ax_deg.set_title("Micro-graph degree distribution")
        ax_deg.set_xlabel("Degree")
        ax_deg.set_ylabel("Count")
        fig_deg.tight_layout()
        deg_hist_path = figures_dir / f"{dataset}_micro_graph_degree_hist.png"
        fig_deg.savefig(deg_hist_path, dpi=150, transparent=True)
        print(f"[INFO] Saved degree histogram to: {deg_hist_path}")
        if args.show_plots:
            fig_deg.show()
        else:
            plt.close(fig_deg)
    except Exception as e:
        print(f"[WARN] Failed to save degree histogram: {e}")

    # Edge feature histograms
    try:
        if pair_extra.size and pair_extra.shape[1] >= 3:
            fig_edge, axes_edge = plt.subplots(1, 3, figsize=(15, 4))
            labels_edge = ["center_dist", "border_min_dist", "log_density_diff"]
            for i in range(3):
                axes_edge[i].hist(pair_extra[:, i], bins=30)
                axes_edge[i].set_title(labels_edge[i])
                axes_edge[i].set_xlabel(labels_edge[i])
                axes_edge[i].set_ylabel("Count")
            fig_edge.tight_layout()
            edge_hist_path = figures_dir / f"{dataset}_edge_features_hist.png"
            fig_edge.savefig(edge_hist_path, dpi=150, transparent=True)
            print(f"[INFO] Saved edge-feature histograms to: {edge_hist_path}")
            if args.show_plots:
                fig_edge.show()
            else:
                plt.close(fig_edge)
    except Exception as e:
        print(f"[WARN] Failed to save edge-feature histograms: {e}")

    # Micro distance heatmap
    try:
        d_use = X_train_use.shape[1]
        centroids_eff = micro_features[:, :d_use]
        D_micro = pairwise_distances(centroids_eff, metric="euclidean")
        fig_heat, ax_heat = plt.subplots(figsize=(6, 5))
        im = ax_heat.imshow(D_micro, interpolation="nearest", aspect="auto")
        fig_heat.colorbar(im, ax=ax_heat)
        ax_heat.set_title("Pairwise micro-centroid distance matrix")
        ax_heat.set_xlabel("Micro index")
        ax_heat.set_ylabel("Micro index")
        fig_heat.tight_layout()
        heat_path = figures_dir / f"{dataset}_micro_distance_matrix.png"
        fig_heat.savefig(heat_path, dpi=150, transparent=True)
        print(f"[INFO] Saved distance matrix heatmap to: {heat_path}")
        if args.show_plots:
            fig_heat.show()
        else:
            plt.close(fig_heat)
    except Exception as e:
        print(f"[WARN] Failed to save distance matrix heatmap: {e}")

    # Micro-graph visualization
    fig_graph = plot_micro_graph(micro_features, micro_major_labels, edge_list, dist_thresh, seed=args.seed)
    graph_path = figures_dir / f"{dataset}_micro_graph.png"
    fig_graph.savefig(graph_path, dpi=150, transparent=True)
    print(f"[INFO] Saved micro-graph figure to: {graph_path}")
    if args.show_plots:
        fig_graph.show()
    else:
        plt.close(fig_graph)

    # Step 5: Train pair model
    model = train_pair_transformer(
        micro_features=micro_features,
        micro_major_labels=micro_major_labels,
        edge_list=edge_list,
        pair_extra=pair_extra,
        device=device,
        args=args,
        dataset=dataset,
        out_dir=out_dir,
        figures_dir=figures_dir,
    )

    # Step 6: Merge to k_target
    micro_to_macro = merge_micro_clusters_learned(
        micro_features=micro_features,
        edge_list=edge_list,
        pair_extra=pair_extra,
        model=model,
        device=device,
        k_target=k_target,
    )

    # Step 7: Assign labels (in the same feature space used for clustering)
    y_pred_train_macro = np.zeros_like(y_train)
    for i in range(len(X_train_use)):
        m_new = train_micro_new[i]
        y_pred_train_macro[i] = micro_to_macro[m_new]

    # Test: nearest effective micro centroid in the same space
    centroids_all = km.cluster_centers_  # centers in X_train_use space
    centroids_eff = centroids_all[new_to_orig]  # effective micro centers
    dtest = pairwise_distances(X_test_use, centroids_eff, metric="euclidean") if len(X_test_use) else np.zeros((0, len(centroids_eff)))
    nearest_micro_eff = dtest.argmin(axis=1) if len(X_test_use) else np.zeros((0,), dtype=int)
    y_pred_test_macro = micro_to_macro[nearest_micro_eff] if len(X_test_use) else np.zeros((0,), dtype=int)

    # Metrics
    nmi_train = normalized_mutual_info_score(y_train, y_pred_train_macro)
    nmi_test = normalized_mutual_info_score(y_test, y_pred_test_macro) if len(y_test) else float("nan")

    ami_train = adjusted_mutual_info_score(y_train, y_pred_train_macro)
    ami_test = adjusted_mutual_info_score(y_test, y_pred_test_macro) if len(y_test) else float("nan")

    ari_train = adjusted_rand_score(y_train, y_pred_train_macro)
    ari_test = adjusted_rand_score(y_test, y_pred_test_macro) if len(y_test) else float("nan")

    fmi_train = fowlkes_mallows_score(y_train, y_pred_train_macro)
    fmi_test = fowlkes_mallows_score(y_test, y_pred_test_macro) if len(y_test) else float("nan")

    acc_train, map_train = clustering_accuracy_with_map(y_train, y_pred_train_macro)
    if len(y_test):
        acc_test, map_test = clustering_accuracy_with_map(y_test, y_pred_test_macro)
    else:
        acc_test, map_test = float("nan"), {}

    # Internal metrics computed on the same space used for clustering (X_*_use)
    sil_train = safe_cluster_metric(silhouette_score, X_train_use, y_pred_train_macro, "silhouette (train)")
    sil_test = safe_cluster_metric(silhouette_score, X_test_use, y_pred_test_macro, "silhouette (test)")

    db_train = safe_cluster_metric(davies_bouldin_score, X_train_use, y_pred_train_macro, "DB (train)")
    db_test = safe_cluster_metric(davies_bouldin_score, X_test_use, y_pred_test_macro, "DB (test)")

    ch_train = safe_cluster_metric(calinski_harabasz_score, X_train_use, y_pred_train_macro, "CH (train)")
    ch_test = safe_cluster_metric(calinski_harabasz_score, X_test_use, y_pred_test_macro, "CH (test)")

    print(f"\n[RESULT] Dataset = {dataset}")
    print(f"[RESULT] Train NMI   = {nmi_train:.4f}, ACC = {acc_train:.4f}")
    print(f"[RESULT] Train AMI   = {ami_train:.4f}, ARI = {ari_train:.4f}, FMI = {fmi_train:.4f}")
    print(f"[RESULT] Train Sil   = {sil_train:.4f}, DB  = {db_train:.4f}, CH  = {ch_train:.4f}")
    print(f"[RESULT] Test  NMI   = {nmi_test:.4f}, ACC = {acc_test:.4f}")
    print(f"[RESULT] Test  AMI   = {ami_test:.4f}, ARI = {ari_test:.4f}, FMI = {fmi_test:.4f}")
    print(f"[RESULT] Test  Sil   = {sil_test:.4f}, DB  = {db_test:.4f}, CH  = {ch_test:.4f}")

    # For test visualization
    if len(y_test):
        y_pred_test_mapped = apply_label_mapping(y_pred_test_macro, map_test)
        correct_test = (y_pred_test_mapped == y_test)
    else:
        correct_test = np.zeros((0,), dtype=bool)

    # ==================== PLOTS: TRAIN + TEST ======================

    # plot in 2D from the used feature space
    Xtr2 = to_2d_for_plot(X_train_use, seed=args.seed)
    Xte2 = to_2d_for_plot(X_test_use, seed=args.seed)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].scatter(Xtr2[:, 0], Xtr2[:, 1], c=y_train, cmap="tab10", s=20, alpha=0.9)
    axes[0].set_title("Train ground truth")
    axes[0].set_xlabel("dim-1"); axes[0].set_ylabel("dim-2")

    axes[1].scatter(Xtr2[:, 0], Xtr2[:, 1], c=y_pred_train_macro, cmap="tab10", s=20, alpha=0.9)
    axes[1].set_title(f"Train predicted (NMI={nmi_train:.3f}, ACC={acc_train:.3f})")
    axes[1].set_xlabel("dim-1"); axes[1].set_ylabel("dim-2")

    axes[2].set_title(f"Test (NMI={nmi_test:.3f}, ACC={acc_test:.3f})")
    axes[2].set_xlabel("dim-1"); axes[2].set_ylabel("dim-2")

    if len(Xte2):
        axes[2].scatter(Xte2[:, 0], Xte2[:, 1], c="blue", s=20, alpha=0.3, label="Test points")
        axes[2].scatter(Xte2[correct_test, 0], Xte2[correct_test, 1], c="green", s=20, alpha=0.9, label="Correct")
        axes[2].scatter(Xte2[~correct_test, 0], Xte2[~correct_test, 1], c="red", s=30, alpha=0.9, marker="x", label="Mis-clustered")
        axes[2].legend(loc="best", fontsize=8)

    fig.tight_layout()
    results_fig_path = figures_dir / f"{dataset}_train_test.png"
    fig.savefig(results_fig_path, dpi=150, transparent=True)
    print(f"[INFO] Saved train/test results figure to: {results_fig_path}")
    if args.show_plots:
        fig.show()
    else:
        plt.close(fig)

    # Full GT plot (2D) - uses PCA-to-2D for viz only
    try:
        # Use original X for GT plot but reduce to 2D for display
        X2_full = to_2d_for_plot(X.astype(np.float32), seed=args.seed)
        fig_full, ax_full = plt.subplots(figsize=(6, 5))
        ax_full.scatter(X2_full[:, 0], X2_full[:, 1], c=y, cmap="tab10", s=15, alpha=0.9)
        ax_full.set_title("Full dataset ground truth")
        ax_full.set_xlabel("dim-1"); ax_full.set_ylabel("dim-2")
        fig_full.tight_layout()
        full_gt_path = figures_dir / f"{dataset}_full_ground_truth.png"
        fig_full.savefig(full_gt_path, dpi=150, transparent=True)
        print(f"[INFO] Saved full dataset ground-truth figure to: {full_gt_path}")
        if args.show_plots:
            fig_full.show()
        else:
            plt.close(fig_full)
    except Exception as e:
        print(f"[WARN] Failed to save full dataset ground-truth plot: {e}")

    # Metric bar plots
    try:
        metric_values = {
            "NMI": (nmi_train, nmi_test),
            "AMI": (ami_train, ami_test),
            "ARI": (ari_train, ari_test),
            "FMI": (fmi_train, fmi_test),
            "Silhouette": (sil_train, sil_test),
            "DaviesBouldin": (db_train, db_test),
            "CalinskiHarabasz": (ch_train, ch_test),
            "ACC": (acc_train, acc_test),
        }
        for name, (v_train, v_test) in metric_values.items():
            fig_m, ax_m = plt.subplots(figsize=(4, 4))
            ax_m.bar(["Train", "Test"], [v_train, v_test])
            ax_m.set_title(f"{name} (train vs test)")
            ax_m.set_ylabel(name)
            fig_m.tight_layout()
            metric_path = figures_dir / f"{dataset}_metric_{name.lower()}.png"
            fig_m.savefig(metric_path, dpi=150, transparent=True)
            if args.show_plots:
                fig_m.show()
            else:
                plt.close(fig_m)
    except Exception as e:
        print(f"[WARN] Failed to save metric bar plots: {e}")

    # Save txt results
    results_txt_path = out_dir / f"{dataset}_results.txt"
    with open(results_txt_path, "w", encoding="utf-8") as f:
        f.write(f"dataset          = {dataset}\n")
        f.write(f"run_id           = {args.run_id}\n")
        f.write(f"seed             = {args.seed}\n")
        f.write(f"split_strategy   = {args.split_strategy}\n")
        f.write(f"train_fraction   = {args.train_fraction}\n")
        f.write("\n")
        f.write(f"reduce_dim       = {args.reduce_dim}\n")
        f.write(f"pca_trigger_dim  = {args.pca_trigger_dim}\n")
        f.write(f"pca_dim          = {args.pca_dim}\n")
        f.write(f"max_density_dim  = {args.max_density_dim}\n")
        f.write(f"used_dim         = {X_train_use.shape[1]}\n")
        f.write("\n")
        f.write(f"n_micro          = {args.n_micro}\n")
        f.write(f"k_target         = {k_target}\n")
        f.write(f"kmeans_n_init    = {args.kmeans_n_init}\n")
        f.write(f"kmeans_max_iter  = {args.kmeans_max_iter}\n")
        f.write(f"dist_quantile    = {args.dist_quantile}\n")
        f.write(f"dist_thresh      = {dist_thresh}\n")
        f.write(f"b_boundary       = {args.b_boundary}\n")
        f.write("\n")
        f.write(f"d_model          = {args.d_model}\n")
        f.write(f"n_heads          = {args.n_heads}\n")
        f.write(f"attn_hidden      = {args.attn_hidden}\n")
        f.write(f"n_transformer_layers = {args.n_transformer_layers}\n")
        f.write(f"train_epochs     = {args.train_epochs}\n")
        f.write(f"lr               = {args.lr}\n")
        f.write(f"weight_decay     = {args.weight_decay}\n")
        f.write("\n")
        f.write(f"Train NMI        = {nmi_train:.6f}\n")
        f.write(f"Train AMI        = {ami_train:.6f}\n")
        f.write(f"Train ARI        = {ari_train:.6f}\n")
        f.write(f"Train FMI        = {fmi_train:.6f}\n")
        f.write(f"Train Silhouette = {sil_train:.6f}\n")
        f.write(f"Train DB         = {db_train:.6f}\n")
        f.write(f"Train CH         = {ch_train:.6f}\n")
        f.write(f"Train ACC        = {acc_train:.6f}\n")
        f.write("\n")
        f.write(f"Test NMI         = {nmi_test:.6f}\n")
        f.write(f"Test AMI         = {ami_test:.6f}\n")
        f.write(f"Test ARI         = {ari_test:.6f}\n")
        f.write(f"Test FMI         = {fmi_test:.6f}\n")
        f.write(f"Test Silhouette  = {sil_test:.6f}\n")
        f.write(f"Test DB          = {db_test:.6f}\n")
        f.write(f"Test CH          = {ch_test:.6f}\n")
        f.write(f"Test ACC         = {acc_test:.6f}\n")

    print(f"[INFO] Saved scalar results to: {results_txt_path}")

    # Append summary row to CSV
    summary_csv_path = Path.cwd() / "results_summary.csv"
    row = {
        # identifiers
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "dataset": dataset,
        "model": MODEL_NAME,
        "run_id": args.run_id,
        "device": str(device),

        # split
        "seed": args.seed,
        "split_strategy": args.split_strategy,
        "train_fraction": args.train_fraction,

        # high-d options
        "reduce_dim": args.reduce_dim,
        "pca_trigger_dim": args.pca_trigger_dim,
        "pca_dim": args.pca_dim,
        "max_density_dim": args.max_density_dim,
        "used_dim": int(X_train_use.shape[1]),

        # configs
        "n_micro": args.n_micro,
        "k_target": k_target,
        "kmeans_n_init": args.kmeans_n_init,
        "kmeans_max_iter": args.kmeans_max_iter,
        "dist_quantile": args.dist_quantile,
        "dist_thresh": float(dist_thresh) if np.isfinite(dist_thresh) else "",
        "b_boundary": args.b_boundary,

        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "attn_hidden": args.attn_hidden,
        "n_transformer_layers": args.n_transformer_layers,
        "train_epochs": args.train_epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,

        # metrics (train)
        "train_nmi": nmi_train,
        "train_ami": ami_train,
        "train_ari": ari_train,
        "train_fmi": fmi_train,
        "train_sil": sil_train,
        "train_db": db_train,
        "train_ch": ch_train,
        "train_acc": acc_train,

        # metrics (test)
        "test_nmi": nmi_test,
        "test_ami": ami_test,
        "test_ari": ari_test,
        "test_fmi": fmi_test,
        "test_sil": sil_test,
        "test_db": db_test,
        "test_ch": ch_test,
        "test_acc": acc_test,

        # artifacts
        "results_txt": str(results_txt_path),
        "figures_dir": str(figures_dir),
        "micro_graph_png": str(graph_path),
        "train_test_png": str(results_fig_path),
    }
    append_row_to_csv(summary_csv_path, row)
    print(f"[INFO] Appended run summary to CSV: {summary_csv_path}")

    return {
        "dataset": dataset,
        "run_id": args.run_id,
        "train_nmi": nmi_train,
        "train_acc": acc_train,
        "test_nmi": nmi_test,
        "test_acc": acc_test,
        "results_txt": str(results_txt_path),
        "micro_graph_png": str(graph_path),
        "train_test_png": str(results_fig_path),
        "figures_dir": str(figures_dir),
        "summary_csv": str(summary_csv_path),
    }


if __name__ == "__main__":
    gothic_main()
