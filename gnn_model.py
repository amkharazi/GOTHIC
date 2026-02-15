"""
gnn_model.py

Simple GNN baseline for clustering: Graph AutoEncoder (GAE) with a 2-layer GCN encoder.
No torch-geometric required.

Pipeline:
  1) Load dataset via loader.load_dataset.
  2) Train/test split (random or density-balanced) for fair evaluation.
  3) Build kNN graph on TRAIN set only.
  4) Train a GCN encoder to reconstruct adjacency (unsupervised).
  5) Get embeddings for TRAIN nodes, fit KMeans (k_target).
  6) Get embeddings for TEST nodes by running the encoder on a TEST-only graph
     (same learned weights, separate graph).
  7) Predict clusters for train & test with KMeans, compute metrics.
  8) Save plots (transparent) + results under out_dir/run_id/.
  9) Append a run summary row to results_summary.csv (in Path.cwd()).

Notes:
  - For high-dimensional datasets, plots use PCA to 2D by default (set --plot_pca).
  - Silhouette/DB/CH are computed in embedding space Z (metric_space="Z").
"""

MODEL_NAME = "gnn"

import csv
import json
from datetime import datetime

import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.cluster import KMeans
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    adjusted_rand_score,
    fowlkes_mallows_score,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Local imports
from helper import set_seed, clustering_accuracy_with_map, apply_label_mapping
from loader import load_dataset


# ======================== DEFAULT CONFIG ===========================

DEFAULT_SEED = 42
DEFAULT_TRAIN_FRACTION = 0.8
DEFAULT_SPLIT_STRATEGY = "balanced"
DEFAULT_DATASET = "compound"

# GNN/GAE defaults
DEFAULT_KNN_K = 10
DEFAULT_HIDDEN_DIM = 64
DEFAULT_EMBED_DIM = 16
DEFAULT_DROPOUT = 0.1
DEFAULT_LR = 1e-2
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_EPOCHS = 200
DEFAULT_LOG_EVERY = 20

DEFAULT_KMEANS_N_INIT = 50
DEFAULT_KMEANS_MAX_ITER = 500

# Dataset-specific default number of clusters
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
    p = argparse.ArgumentParser(description="Simple GNN (GCN-GAE) baseline for clustering.")

    # Core
    p.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--train_fraction", type=float, default=DEFAULT_TRAIN_FRACTION)
    p.add_argument("--split_strategy", type=str, default=DEFAULT_SPLIT_STRATEGY, choices=["random", "balanced"])
    p.add_argument("--data_root", type=str, default="datasets")
    p.add_argument("--out_dir", type=str, default="outputs")
    p.add_argument("--run_id", type=str, default="gnn_default")

    # Target clusters
    p.add_argument("--k_target", type=int, default=-1)

    # Graph
    p.add_argument("--knn_k", type=int, default=DEFAULT_KNN_K, help="k for kNN graph construction.")

    # GNN
    p.add_argument("--hidden_dim", type=int, default=DEFAULT_HIDDEN_DIM)
    p.add_argument("--embed_dim", type=int, default=DEFAULT_EMBED_DIM)
    p.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--log_every", type=int, default=DEFAULT_LOG_EVERY)

    # KMeans on embeddings
    p.add_argument("--kmeans_n_init", type=int, default=DEFAULT_KMEANS_N_INIT)
    p.add_argument("--kmeans_max_iter", type=int, default=DEFAULT_KMEANS_MAX_ITER)

    # Plotting
    p.add_argument("--plot_pca", action="store_true", help="Use PCA to 2D for plots (recommended for high-dim).")
    p.add_argument("--show_plots", action="store_true")

    # Device
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    return p.parse_args(argv)


# =========================== CSV SUMMARY (same as spectral) ============================

def _to_csv_value(v):
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, Path):
        return str(v)
    return v


def append_row_to_csv(csv_path: Path, row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    row = {k: _to_csv_value(v) for k, v in row.items()}

    if not csv_path.exists():
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        return

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


# =========================== UTILITIES =============================

def train_test_split_with_strategy(X, y, train_frac=0.8, strategy="balanced", seed=42):
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    indices = np.arange(n)

    if strategy == "random":
        perm = rng.permutation(indices)
        n_train = int(round(train_frac * n))
        train_idx = perm[:n_train]
        test_idx = perm[n_train:]
    elif strategy == "balanced":
        train_idx = []
        test_idx = []
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

            order = np.argsort(-densities)  # densest first
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
        train_idx, test_idx,
    )


def safe_cluster_metric(fn, X_data, labels_pred, name: str) -> float:
    try:
        if len(np.unique(labels_pred)) < 2:
            return float("nan")
        return float(fn(X_data, labels_pred))
    except Exception as e:
        print(f"[WARN] Failed to compute {name}: {e}")
        return float("nan")


def build_knn_adjacency(X: np.ndarray, k: int) -> np.ndarray:
    """
    Build an undirected binary kNN adjacency matrix (no self-loops yet).
    """
    n = X.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)

    k_eff = min(max(1, int(k)), max(1, n - 1))
    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean")
    nn.fit(X)
    _, idx = nn.kneighbors(X)

    A = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in idx[i, 1:]:
            A[i, j] = 1.0
            A[j, i] = 1.0
    return A


def normalize_adjacency(A: np.ndarray, add_self_loops: bool = True) -> np.ndarray:
    """
    Symmetric normalization:  A_hat = D^{-1/2} (A + I) D^{-1/2}
    """
    n = A.shape[0]
    A_hat = A.copy()
    if add_self_loops and n > 0:
        A_hat += np.eye(n, dtype=np.float32)

    deg = A_hat.sum(axis=1)
    deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
    deg_inv_sqrt[~np.isfinite(deg_inv_sqrt)] = 0.0

    A_norm = (deg_inv_sqrt[:, None] * A_hat) * deg_inv_sqrt[None, :]
    return A_norm.astype(np.float32)


def to_2d_for_plot(X: np.ndarray, use_pca: bool) -> np.ndarray:
    if X.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if X.shape[1] >= 2 and not use_pca:
        return X[:, :2]
    if X.shape[1] == 1:
        return np.column_stack([X[:, 0], np.zeros_like(X[:, 0])])
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(X)


# =========================== SIMPLE GCN ============================

class GCNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, embed_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        H = A_norm @ X
        H = self.fc1(H)
        H = F.relu(H)
        H = self.dropout(H)
        Z = A_norm @ H
        Z = self.fc2(Z)
        return Z


def gae_loss_from_logits(logits: torch.Tensor, target_adj: torch.Tensor) -> torch.Tensor:
    """
    BCE loss for adjacency reconstruction, ignoring diagonal.
    Uses pos_weight to reduce class imbalance.
    """
    n = logits.shape[0]
    if n == 0:
        return torch.tensor(0.0, device=logits.device)

    mask = ~torch.eye(n, dtype=torch.bool, device=logits.device)
    logits_v = logits[mask]
    target_v = target_adj[mask]

    pos = target_v.sum().clamp(min=1.0)
    neg = (target_v.numel() - pos).clamp(min=1.0)
    pos_weight = (neg / pos)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return loss_fn(logits_v, target_v)


def train_gae(
    X_np: np.ndarray,
    A_bin_np: np.ndarray,
    hidden_dim: int,
    embed_dim: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    epochs: int,
    log_every: int,
    device: torch.device,
) -> Tuple[GCNEncoder, np.ndarray]:
    """
    Train GAE on a given graph (X, A). Returns trained model and embeddings.
    """
    X = torch.tensor(X_np, dtype=torch.float32, device=device)
    A_bin = torch.tensor(A_bin_np, dtype=torch.float32, device=device)
    A_norm = torch.tensor(normalize_adjacency(A_bin_np, add_self_loops=True), dtype=torch.float32, device=device)

    model = GCNEncoder(in_dim=X.shape[1], hidden_dim=hidden_dim, embed_dim=embed_dim, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        Z = model(X, A_norm)
        logits = Z @ Z.t()
        loss = gae_loss_from_logits(logits, A_bin)
        loss.backward()
        opt.step()

        if log_every > 0 and (ep == 1 or ep % log_every == 0 or ep == epochs):
            print(f"[GNN] epoch {ep:4d}/{epochs} | loss={loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        Z = model(X, A_norm).detach().cpu().numpy()

    return model, Z


def infer_embeddings(model: GCNEncoder, X_np: np.ndarray, A_bin_np: np.ndarray, device: torch.device) -> np.ndarray:
    X = torch.tensor(X_np, dtype=torch.float32, device=device)
    A_norm = torch.tensor(normalize_adjacency(A_bin_np, add_self_loops=True), dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        Z = model(X, A_norm).detach().cpu().numpy()
    return Z


# ============================= MAIN ================================

def gnn_main(args=None) -> Dict[str, Any]:
    if args is None:
        args = parse_args()

    dataset = args.dataset.lower()
    set_seed(args.seed)

    # device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")

    data_root = Path(args.data_root)
    root_out_dir = Path(args.out_dir)
    out_dir = root_out_dir / args.run_id
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
        k_target = args.k_target
        print(f"[INFO] Using user-specified k_target={k_target}")

    # Load data
    X, y = load_dataset(dataset, base_dir=str(data_root))
    X = X.astype(np.float32)
    y = y.astype(int)
    print(f"[INFO] Loaded dataset '{dataset}': X.shape={X.shape}, unique labels={np.unique(y)}")

    # Split for evaluation
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split_with_strategy(
        X, y,
        train_frac=args.train_fraction,
        strategy=args.split_strategy,
        seed=args.seed,
    )
    print(f"[INFO] Train size: {len(X_train)}, Test size: {len(X_test)}")

    # ----------------- TRAIN GRAPH + GAE TRAINING ------------------
    print(f"[INFO] Building TRAIN kNN graph (k={args.knn_k}) ...")
    A_train = build_knn_adjacency(X_train, k=args.knn_k)

    print("[INFO] Training GCN-GAE (unsupervised adjacency reconstruction) ...")
    model, Z_train = train_gae(
        X_np=X_train,
        A_bin_np=A_train,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        log_every=args.log_every,
        device=device,
    )

    # ----------------- INFER TEST EMBEDDINGS -----------------------
    print(f"[INFO] Building TEST kNN graph (k={args.knn_k}) for embedding inference ...")
    if len(X_test) >= 2:
        A_test = build_knn_adjacency(X_test, k=min(args.knn_k, len(X_test) - 1))
    else:
        A_test = np.zeros((len(X_test), len(X_test)), dtype=np.float32)

    Z_test = infer_embeddings(model, X_test, A_test, device=device)

    # ----------------- CLUSTER EMBEDDINGS --------------------------
    km = KMeans(
        n_clusters=k_target,
        n_init=args.kmeans_n_init,
        max_iter=args.kmeans_max_iter,
        random_state=args.seed,
    )
    y_pred_train = km.fit_predict(Z_train).astype(int)
    y_pred_test = km.predict(Z_test).astype(int) if len(Z_test) > 0 else np.array([], dtype=int)

    # ==================== METRICS =================================
    nmi_train = normalized_mutual_info_score(y_train, y_pred_train)
    ami_train = adjusted_mutual_info_score(y_train, y_pred_train)
    ari_train = adjusted_rand_score(y_train, y_pred_train)
    fmi_train = fowlkes_mallows_score(y_train, y_pred_train)
    acc_train, map_train = clustering_accuracy_with_map(y_train, y_pred_train)

    if len(y_test) > 0 and len(y_pred_test) == len(y_test):
        nmi_test = normalized_mutual_info_score(y_test, y_pred_test)
        ami_test = adjusted_mutual_info_score(y_test, y_pred_test)
        ari_test = adjusted_rand_score(y_test, y_pred_test)
        fmi_test = fowlkes_mallows_score(y_test, y_pred_test)
        acc_test, map_test = clustering_accuracy_with_map(y_test, y_pred_test)
    else:
        nmi_test = ami_test = ari_test = fmi_test = acc_test = float("nan")
        map_test = {}

    # Embedding-space cluster validity
    sil_train = safe_cluster_metric(silhouette_score, Z_train, y_pred_train, "silhouette (train, Z)")
    sil_test = safe_cluster_metric(silhouette_score, Z_test, y_pred_test, "silhouette (test, Z)")
    db_train_val = safe_cluster_metric(davies_bouldin_score, Z_train, y_pred_train, "Davies-Bouldin (train, Z)")
    db_test_val = safe_cluster_metric(davies_bouldin_score, Z_test, y_pred_test, "Davies-Bouldin (test, Z)")
    ch_train = safe_cluster_metric(calinski_harabasz_score, Z_train, y_pred_train, "Calinski-Harabasz (train, Z)")
    ch_test = safe_cluster_metric(calinski_harabasz_score, Z_test, y_pred_test, "Calinski-Harabasz (test, Z)")

    print(f"[RESULT] Dataset = {dataset} (GNN-GAE + KMeans)")
    print(f"[RESULT] Train NMI   = {nmi_train:.4f}, ACC = {acc_train:.4f}")
    print(f"[RESULT] Train AMI   = {ami_train:.4f}, ARI = {ari_train:.4f}, FMI = {fmi_train:.4f}")
    print(f"[RESULT] Train Sil(Z)= {sil_train:.4f}, DB(Z)= {db_train_val:.4f}, CH(Z)= {ch_train:.4f}")
    print(f"[RESULT] Test  NMI   = {nmi_test:.4f}, ACC = {acc_test:.4f}")
    print(f"[RESULT] Test  AMI   = {ami_test:.4f}, ARI = {ari_test:.4f}, FMI = {fmi_test:.4f}")
    print(f"[RESULT] Test  Sil(Z)= {sil_test:.4f}, DB(Z)= {db_test_val:.4f}, CH(Z)= {ch_test:.4f}")

    # For visualization correctness
    if len(y_test) > 0 and len(y_pred_test) == len(y_test):
        y_pred_test_mapped = apply_label_mapping(y_pred_test, map_test)
        correct_test = (y_pred_test_mapped == y_test)
    else:
        correct_test = np.array([], dtype=bool)

    # ==================== PLOTS (spectral-like) ====================
    X_train_2d = to_2d_for_plot(X_train, use_pca=args.plot_pca or (X_train.shape[1] > 2))
    X_test_2d = to_2d_for_plot(X_test, use_pca=args.plot_pca or (X_train.shape[1] > 2))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train, cmap="tab10", s=20, alpha=0.9)
    ax.set_title("Train ground truth")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax = axes[1]
    ax.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_pred_train, cmap="tab10", s=20, alpha=0.9)
    ax.set_title(f"Train GNN (NMI={nmi_train:.3f}, ACC={acc_train:.3f})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax = axes[2]
    ax.set_title(f"Test GNN (NMI={nmi_test:.3f}, ACC={acc_test:.3f})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if len(X_test_2d) > 0:
        ax.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c="blue", s=20, alpha=0.3, label="Test points")
        if len(correct_test) == len(X_test_2d):
            ax.scatter(X_test_2d[correct_test, 0], X_test_2d[correct_test, 1], c="green", s=20, alpha=0.9, label="Correct")
            ax.scatter(X_test_2d[~correct_test, 0], X_test_2d[~correct_test, 1], c="red", s=30, alpha=0.9, marker="x", label="Mis-clustered")
        ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    train_test_path = figures_dir / f"{dataset}_gnn_train_test.png"
    fig.savefig(train_test_path, dpi=150, transparent=True)
    print(f"[INFO] Saved GNN train/test figure to: {train_test_path}")
    if args.show_plots:
        plt.show()
    else:
        plt.close(fig)

    # Full dataset GT plot (optional)
    try:
        X_2d = to_2d_for_plot(X, use_pca=args.plot_pca or (X.shape[1] > 2))
        fig_full, ax_full = plt.subplots(figsize=(6, 5))
        ax_full.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="tab10", s=15, alpha=0.9)
        ax_full.set_title("Full dataset ground truth")
        ax_full.set_xlabel("x")
        ax_full.set_ylabel("y")
        fig_full.tight_layout()

        full_gt_path = figures_dir / f"{dataset}_gnn_full_ground_truth.png"
        fig_full.savefig(full_gt_path, dpi=150, transparent=True)
        print(f"[INFO] Saved full dataset GT figure (GNN run) to: {full_gt_path}")
        if args.show_plots:
            plt.show()
        else:
            plt.close(fig_full)
    except Exception as e:
        print(f"[WARN] Failed to save full dataset ground-truth plot: {e}")

    # Save scalar results (spectral-like formatting)
    results_txt_path = out_dir / f"{dataset}_gnn_results.txt"
    with open(results_txt_path, "w", encoding="utf-8") as f:
        f.write(f"dataset          = {dataset}\n")
        f.write(f"model            = {MODEL_NAME}\n")
        f.write(f"run_id           = {args.run_id}\n")
        f.write(f"seed             = {args.seed}\n")
        f.write(f"device           = {device}\n")
        f.write(f"split_strategy   = {args.split_strategy}\n")
        f.write(f"train_fraction   = {args.train_fraction}\n")
        f.write(f"k_target         = {k_target}\n")
        f.write("\n")
        f.write(f"knn_k            = {args.knn_k}\n")
        f.write(f"hidden_dim       = {args.hidden_dim}\n")
        f.write(f"embed_dim        = {args.embed_dim}\n")
        f.write(f"dropout          = {args.dropout}\n")
        f.write(f"lr               = {args.lr}\n")
        f.write(f"weight_decay     = {args.weight_decay}\n")
        f.write(f"epochs           = {args.epochs}\n")
        f.write(f"kmeans_n_init    = {args.kmeans_n_init}\n")
        f.write(f"kmeans_max_iter  = {args.kmeans_max_iter}\n")
        f.write(f"metric_space     = Z\n")
        f.write("\n")
        f.write(f"Train NMI        = {nmi_train:.6f}\n")
        f.write(f"Train AMI        = {ami_train:.6f}\n")
        f.write(f"Train ARI        = {ari_train:.6f}\n")
        f.write(f"Train FMI        = {fmi_train:.6f}\n")
        f.write(f"Train Sil(Z)     = {sil_train:.6f}\n")
        f.write(f"Train DB(Z)      = {db_train_val:.6f}\n")
        f.write(f"Train CH(Z)      = {ch_train:.6f}\n")
        f.write(f"Train ACC        = {acc_train:.6f}\n")
        f.write("\n")
        f.write(f"Test NMI         = {nmi_test:.6f}\n")
        f.write(f"Test AMI         = {ami_test:.6f}\n")
        f.write(f"Test ARI         = {ari_test:.6f}\n")
        f.write(f"Test FMI         = {fmi_test:.6f}\n")
        f.write(f"Test Sil(Z)      = {sil_test:.6f}\n")
        f.write(f"Test DB(Z)       = {db_test_val:.6f}\n")
        f.write(f"Test CH(Z)       = {ch_test:.6f}\n")
        f.write(f"Test ACC         = {acc_test:.6f}\n")

    print(f"[INFO] Saved GNN scalar results to: {results_txt_path}")

    # Append run summary (same as spectral keys)
    summary_csv_path = Path.cwd() / "results_summary.csv"
    row = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "dataset": dataset,
        "model": MODEL_NAME,
        "run_id": args.run_id,
        "seed": args.seed,
        "split_strategy": args.split_strategy,
        "train_fraction": args.train_fraction,
        "k_target": k_target,

        # model-specific params
        "knn_k": args.knn_k,
        "hidden_dim": args.hidden_dim,
        "embed_dim": args.embed_dim,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "kmeans_n_init": args.kmeans_n_init,
        "kmeans_max_iter": args.kmeans_max_iter,
        "metric_space": "Z",

        # metrics (same column names as spectral)
        "train_nmi": nmi_train,
        "train_ami": ami_train,
        "train_ari": ari_train,
        "train_fmi": fmi_train,
        "train_sil": sil_train,
        "train_db": db_train_val,
        "train_ch": ch_train,
        "train_acc": acc_train,

        "test_nmi": nmi_test,
        "test_ami": ami_test,
        "test_ari": ari_test,
        "test_fmi": fmi_test,
        "test_sil": sil_test,
        "test_db": db_test_val,
        "test_ch": ch_test,
        "test_acc": acc_test,

        "results_txt": str(results_txt_path),
        "figures_dir": str(figures_dir),
        "train_test_png": str(train_test_path),
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
        "figures_dir": str(figures_dir),
        "train_test_png": str(train_test_path),
        "summary_csv": str(summary_csv_path),
    }


if __name__ == "__main__":
    gnn_main()
