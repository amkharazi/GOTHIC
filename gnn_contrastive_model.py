"""
gnn_contrastive_model.py

Modern GNN variant for clustering:
  GRACE-style Graph Contrastive Learning (GCN encoder) + KMeans.

No torch-geometric required.

Pipeline:
  1) Load dataset (X,y) via loader.load_dataset
  2) Train/test split (random or density-balanced)
  3) Build kNN graph on TRAIN features
  4) Train GCN using contrastive loss across two augmented graph views:
       - edge dropout
       - feature dropout
     positives: same node across views
     negatives: other nodes in batch (full graph)
  5) Embed TRAIN and TEST (using learned encoder)
  6) KMeans on TRAIN embeddings -> y_pred_train, predict TEST via kmeans.predict
  7) Evaluate + plots + results txt
  8) Append a run summary row to results_summary.csv (in Path.cwd())

Run:
  python gnn_contrastive_model.py --dataset digits --run_id gcl_digits --k_target 10 --plot_pca
"""

MODEL_NAME = "gnn_contrastive"

import csv
import json
from datetime import datetime

import argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    adjusted_rand_score,
    fowlkes_mallows_score,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

from loader import load_dataset
from helper import set_seed, clustering_accuracy_with_map, apply_label_mapping


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
    p = argparse.ArgumentParser(description="GNN Contrastive (GRACE-style) clustering baseline.")

    p.add_argument("--dataset", type=str, default="compound")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_fraction", type=float, default=0.8)
    p.add_argument("--split_strategy", type=str, default="balanced", choices=["random", "balanced"])
    p.add_argument("--data_root", type=str, default="datasets")
    p.add_argument("--out_dir", type=str, default="outputs")
    p.add_argument("--run_id", type=str, default="gcl_default")

    p.add_argument("--k_target", type=int, default=-1)

    # graph
    p.add_argument("--knn_k", type=int, default=10, help="k for kNN graph construction.")

    # gcn
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--embed_dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)

    # contrastive training
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=0.2)

    # augmentations
    p.add_argument("--edge_drop", type=float, default=0.2, help="Probability of dropping an edge in each view.")
    p.add_argument("--feat_drop", type=float, default=0.2, help="Probability of dropping each feature entry in each view.")

    # kmeans
    p.add_argument("--kmeans_n_init", type=int, default=50)
    p.add_argument("--kmeans_max_iter", type=int, default=500)

    # plots
    p.add_argument("--plot_pca", action="store_true")
    p.add_argument("--show_plots", action="store_true")

    # device
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


# =========================== HELPERS ============================

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
        train_idx, test_idx = [], []
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

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx], train_idx, test_idx


def safe_cluster_metric(fn, X_data, labels_pred, name: str) -> float:
    try:
        if len(np.unique(labels_pred)) < 2:
            return float("nan")
        return float(fn(X_data, labels_pred))
    except Exception as e:
        print(f"[WARN] Failed to compute {name}: {e}")
        return float("nan")


def to_2d_for_plot(X: np.ndarray, use_pca: bool) -> np.ndarray:
    if X.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if X.shape[1] >= 2 and not use_pca:
        return X[:, :2]
    if X.shape[1] == 1:
        return np.column_stack([X[:, 0], np.zeros_like(X[:, 0])])
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(X)


def build_knn_adjacency(X: np.ndarray, k: int) -> np.ndarray:
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
    n = A.shape[0]
    A_hat = A.copy()
    if add_self_loops and n > 0:
        A_hat += np.eye(n, dtype=np.float32)

    deg = A_hat.sum(axis=1)
    deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
    deg_inv_sqrt[~np.isfinite(deg_inv_sqrt)] = 0.0

    A_norm = (deg_inv_sqrt[:, None] * A_hat) * deg_inv_sqrt[None, :]
    return A_norm.astype(np.float32)


def drop_edges(A: torch.Tensor, drop_prob: float) -> torch.Tensor:
    if drop_prob <= 0 or A.numel() == 0:
        return A
    n = A.shape[0]
    if n <= 1:
        return A

    A2 = A.clone()
    triu = torch.triu(torch.ones((n, n), device=A.device, dtype=torch.bool), diagonal=1)
    edges = (A2 > 0) & triu
    rand = torch.rand((n, n), device=A.device)
    drop = (rand < drop_prob) & edges
    A2[drop] = 0.0
    A2 = torch.maximum(A2, A2.t())
    return A2


def drop_features(X: torch.Tensor, drop_prob: float) -> torch.Tensor:
    if drop_prob <= 0:
        return X
    mask = (torch.rand_like(X) > drop_prob).float()
    return X * mask


def info_nce(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    n = z1.shape[0]
    sim = (z1 @ z2.t()) / max(1e-8, float(temperature))
    labels = torch.arange(n, device=z1.device)
    return F.cross_entropy(sim, labels)


# =========================== GCN ============================

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


@torch.no_grad()
def embed(model: nn.Module, X_np: np.ndarray, A_bin_np: np.ndarray, device: torch.device, embed_dim: int) -> np.ndarray:
    if X_np.shape[0] == 0:
        return np.zeros((0, embed_dim), dtype=np.float32)
    X = torch.tensor(X_np, dtype=torch.float32, device=device)
    A_norm = torch.tensor(normalize_adjacency(A_bin_np, add_self_loops=True), dtype=torch.float32, device=device)
    model.eval()
    Z = model(X, A_norm)
    return Z.detach().cpu().numpy()


# ============================= MAIN ================================

def gnn_contrastive_main(args=None) -> Dict[str, Any]:
    if args is None:
        args = parse_args()

    set_seed(args.seed)
    dataset = args.dataset.lower()

    # device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")

    root_out_dir = Path(args.out_dir)
    out_dir = root_out_dir / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Outputs will be saved under: {out_dir}")
    print(f"[INFO] Figures will be saved under: {figures_dir}")

    # k_target
    if args.k_target <= 0:
        k_target = DEFAULT_K_TARGETS.get(dataset, 6)
        print(f"[INFO] Using dataset-specific k_target={k_target}")
    else:
        k_target = args.k_target
        print(f"[INFO] Using user-specified k_target={k_target}")

    # load
    X, y = load_dataset(dataset, base_dir=args.data_root)
    X = X.astype(np.float32)
    y = y.astype(int)

    # split
    X_train, X_test, y_train, y_test, _, _ = train_test_split_with_strategy(
        X, y, train_frac=args.train_fraction, strategy=args.split_strategy, seed=args.seed
    )
    print(f"[INFO] Train size={len(X_train)} Test size={len(X_test)} dim={X_train.shape[1]}")

    # build TRAIN graph
    A_train = build_knn_adjacency(X_train, k=args.knn_k)
    A_train_t = torch.tensor(A_train, dtype=torch.float32, device=device)
    Xtr_t = torch.tensor(X_train, dtype=torch.float32, device=device)

    # model
    model = GCNEncoder(X_train.shape[1], args.hidden_dim, args.embed_dim, args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # train contrastive
    print("[INFO] Training graph contrastive encoder (GRACE-style) ...")
    model.train()
    for ep in range(1, args.epochs + 1):
        # two augmented views
        A1 = drop_edges(A_train_t, args.edge_drop)
        A2 = drop_edges(A_train_t, args.edge_drop)

        X1 = drop_features(Xtr_t, args.feat_drop)
        X2 = drop_features(Xtr_t, args.feat_drop)

        A1n = torch.tensor(normalize_adjacency(A1.detach().cpu().numpy(), add_self_loops=True), dtype=torch.float32, device=device)
        A2n = torch.tensor(normalize_adjacency(A2.detach().cpu().numpy(), add_self_loops=True), dtype=torch.float32, device=device)

        z1 = model(X1, A1n)
        z2 = model(X2, A2n)

        loss = info_nce(z1, z2, args.temperature)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if ep == 1 or ep % 50 == 0 or ep == args.epochs:
            print(f"[GCL] epoch {ep:4d}/{args.epochs} | loss={loss.item():.6f}")

    # embed train/test
    Z_train = embed(model, X_train, A_train, device=device, embed_dim=args.embed_dim)

    if len(X_test) >= 2:
        A_test = build_knn_adjacency(X_test, k=min(args.knn_k, len(X_test) - 1))
    else:
        A_test = np.zeros((len(X_test), len(X_test)), dtype=np.float32)
    Z_test = embed(model, X_test, A_test, device=device, embed_dim=args.embed_dim)

    # KMeans clustering on embeddings
    km = KMeans(
        n_clusters=k_target,
        n_init=args.kmeans_n_init,
        max_iter=args.kmeans_max_iter,
        random_state=args.seed,
    )
    y_pred_train = km.fit_predict(Z_train).astype(int)
    y_pred_test = km.predict(Z_test).astype(int) if len(Z_test) else np.array([], dtype=int)

    # metrics
    nmi_train = normalized_mutual_info_score(y_train, y_pred_train)
    ami_train = adjusted_mutual_info_score(y_train, y_pred_train)
    ari_train = adjusted_rand_score(y_train, y_pred_train)
    fmi_train = fowlkes_mallows_score(y_train, y_pred_train)
    acc_train, map_train = clustering_accuracy_with_map(y_train, y_pred_train)

    if len(y_test) > 0:
        nmi_test = normalized_mutual_info_score(y_test, y_pred_test)
        ami_test = adjusted_mutual_info_score(y_test, y_pred_test)
        ari_test = adjusted_rand_score(y_test, y_pred_test)
        fmi_test = fowlkes_mallows_score(y_test, y_pred_test)
        acc_test, map_test = clustering_accuracy_with_map(y_test, y_pred_test)
        y_pred_test_mapped = apply_label_mapping(y_pred_test, map_test)
        correct_test = (y_pred_test_mapped == y_test)
    else:
        nmi_test = ami_test = ari_test = fmi_test = acc_test = float("nan")
        map_test = {}
        correct_test = np.array([], dtype=bool)

    # embedding-space cluster validity
    sil_train = safe_cluster_metric(silhouette_score, Z_train, y_pred_train, "silhouette(train,Z)")
    sil_test = safe_cluster_metric(silhouette_score, Z_test, y_pred_test, "silhouette(test,Z)")
    db_train_val = safe_cluster_metric(davies_bouldin_score, Z_train, y_pred_train, "db(train,Z)")
    db_test_val = safe_cluster_metric(davies_bouldin_score, Z_test, y_pred_test, "db(test,Z)")
    ch_train = safe_cluster_metric(calinski_harabasz_score, Z_train, y_pred_train, "ch(train,Z)")
    ch_test = safe_cluster_metric(calinski_harabasz_score, Z_test, y_pred_test, "ch(test,Z)")

    print(f"\n[RESULT] Dataset = {dataset} (GCL-GCN + KMeans)")
    print(f"[RESULT] Train NMI={nmi_train:.4f} ACC={acc_train:.4f} AMI={ami_train:.4f} ARI={ari_train:.4f} FMI={fmi_train:.4f}")
    print(f"[RESULT] Train Sil(Z)={sil_train:.4f} DB(Z)={db_train_val:.4f} CH(Z)={ch_train:.4f}")
    print(f"[RESULT] Test  NMI={nmi_test:.4f} ACC={acc_test:.4f} AMI={ami_test:.4f} ARI={ari_test:.4f} FMI={fmi_test:.4f}")
    print(f"[RESULT] Test  Sil(Z)={sil_test:.4f} DB(Z)={db_test_val:.4f} CH(Z)={ch_test:.4f}")

    # plots in input space
    Xtr2 = to_2d_for_plot(X_train, use_pca=args.plot_pca or (X_train.shape[1] > 2))
    Xte2 = to_2d_for_plot(X_test, use_pca=args.plot_pca or (X_train.shape[1] > 2))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].scatter(Xtr2[:, 0], Xtr2[:, 1], c=y_train, cmap="tab10", s=20, alpha=0.9)
    axes[0].set_title("Train ground truth")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")

    axes[1].scatter(Xtr2[:, 0], Xtr2[:, 1], c=y_pred_train, cmap="tab10", s=20, alpha=0.9)
    axes[1].set_title(f"Train GCL-GNN (NMI={nmi_train:.3f}, ACC={acc_train:.3f})")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("y")

    axes[2].set_title(f"Test GCL-GNN (NMI={nmi_test:.3f}, ACC={acc_test:.3f})")
    axes[2].set_xlabel("x"); axes[2].set_ylabel("y")
    if len(Xte2) > 0:
        axes[2].scatter(Xte2[:, 0], Xte2[:, 1], c="blue", s=20, alpha=0.3, label="Test")
        if len(correct_test) == len(Xte2):
            axes[2].scatter(Xte2[correct_test, 0], Xte2[correct_test, 1], c="green", s=20, alpha=0.9, label="Correct")
            axes[2].scatter(Xte2[~correct_test, 0], Xte2[~correct_test, 1], c="red", s=30, alpha=0.9, marker="x", label="Wrong")
        axes[2].legend(loc="best", fontsize=8)

    fig.tight_layout()
    train_test_path = figures_dir / f"{dataset}_gnn_contrastive_train_test.png"
    fig.savefig(train_test_path, dpi=150, transparent=True)
    print(f"[INFO] Saved train/test plot to: {train_test_path}")
    if args.show_plots:
        plt.show()
    else:
        plt.close(fig)

    # save results txt (spectral-like formatting)
    results_txt_path = out_dir / f"{dataset}_gnn_contrastive_results.txt"
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
        f.write(f"epochs           = {args.epochs}\n")
        f.write(f"lr               = {args.lr}\n")
        f.write(f"weight_decay     = {args.weight_decay}\n")
        f.write(f"temperature      = {args.temperature}\n")
        f.write(f"edge_drop        = {args.edge_drop}\n")
        f.write(f"feat_drop        = {args.feat_drop}\n")
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

    print(f"[INFO] Saved results to: {results_txt_path}")

    # append summary row (same metric keys as spectral)
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
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "temperature": args.temperature,
        "edge_drop": args.edge_drop,
        "feat_drop": args.feat_drop,
        "kmeans_n_init": args.kmeans_n_init,
        "kmeans_max_iter": args.kmeans_max_iter,
        "metric_space": "Z",

        # metrics (match spectral columns)
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
    gnn_contrastive_main()
