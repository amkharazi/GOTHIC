"""
insdpc_model.py

INSDPC baseline: Interactive Neighbors Similarity-based Density Peaks Clustering
(Shihu Liu et al., AIMS Mathematics, 2025) implemented in the same style
as the GOTHIC / KMeans / DBSCAN / HDBSCAN baselines.

Pipeline:
  1) Load dataset via loader.load_dataset.
  2) Train/test split (random or density-balanced).
  3) Run INSDPC on the TRAIN set:
       - Build distance matrix on X_train.
       - For each point, compute its k-nearest neighbors (KNN).
       - For each pair (i,j) in train, compute interactive neighbors similarity Ins(i,j).
       - Define local density rho_i = sum_j Ins(i,j).
       - Define relative distance delta_i (DPC-style, w.r.t. rho).
       - Decision value gamma_i = rho_i * delta_i.
       - Take top K = k_target points by gamma as cluster centers.
       - Two-step assignment (core points via Ins threshold, then remaining points via neighbor voting).
       -> get y_pred_train.
  4) Assign test labels via 1-NN from train: y_pred_test.
  5) Compute metrics:
       - NMI, AMI, ARI, FMI, ACC
       - Silhouette, Davies–Bouldin, Calinski–Harabasz
  6) Save plots (transparent) and scalar results under out_dir/run_id/.
"""

MODEL_NAME = "insdpc"

import csv
import json
from datetime import datetime

import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import matplotlib.pyplot as plt

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

# Local imports
from helper import (
    set_seed,
    clustering_accuracy_with_map,
    apply_label_mapping,
)
from loader import load_dataset


# ======================== DEFAULT CONFIG ===========================

DEFAULT_SEED = 42
DEFAULT_TRAIN_FRACTION = 0.8
DEFAULT_SPLIT_STRATEGY = "balanced"  # "random" or "balanced"
DEFAULT_DATASET = "compound"

DEFAULT_K_NEIGHBORS = 15   # k for KNN / SNN in INSDPC


# =========================== ARG PARSER ============================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="INSDPC baseline for clustering.")

    # Core
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help="Dataset name (synthetic or real) as defined in dataset/loader.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train_fraction", type=float, default=DEFAULT_TRAIN_FRACTION)
    parser.add_argument(
        "--split_strategy",
        type=str,
        default=DEFAULT_SPLIT_STRATEGY,
        choices=["random", "balanced"],
        help="Train/test split strategy.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="datasets",
        help="Base directory where datasets are stored.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs",
        help="Base directory to save results and plots.",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default="insdpc_default",
        help="Run identifier; outputs go under out_dir/run_id/.",
    )

    # INSDPC params
    parser.add_argument(
        "--k_neighbors",
        type=int,
        default=DEFAULT_K_NEIGHBORS,
        help="Number of neighbors k used in KNN / SNN for INSDPC.",
    )
    parser.add_argument(
        "--k_target",
        type=int,
        required=True,
        help="Number of clusters K to output (used to pick K centers).",
    )

    # Misc
    parser.add_argument(
        "--show_plots",
        action="store_true",
        help="If set, show matplotlib figures interactively after saving.",
    )

    return parser.parse_args(argv)


# =========================== UTILITIES =============================

def _to_csv_value(v):
    """Convert values to something safe/flat for CSV."""
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, (Path,)):
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

    # Read existing header
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
        # Upgrade header: read all rows then rewrite
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

    # Append normally
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writerow({k: row.get(k, "") for k in header})

def train_test_split_with_strategy(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float = 0.8,
    strategy: str = "balanced",
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Same behavior as in gothic_model/kmeans_model/dbscan_model/hdbscan_model:
      - 'random'   : simple random split.
      - 'balanced' : per-class, density-aware split (interior points more likely to go to test,
                     boundary points kept in train).
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

            # k-NN within cluster for a rough density estimate
            k_nn = min(10, n_c - 1)
            nn = NearestNeighbors(n_neighbors=k_nn + 1)
            nn.fit(X_c)
            dists, _ = nn.kneighbors(X_c)
            # ignore self distance
            mean_dists = dists[:, 1:].mean(axis=1)
            densities = 1.0 / (eps + mean_dists)

            # higher "density" -> more interior -> send more to test
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
        train_idx, test_idx,
    )


def safe_cluster_metric(fn, X_data, labels_pred, name: str) -> float:
    """
    Compute an internal clustering metric, but handle edge cases gracefully.
    """
    try:
        unique = np.unique(labels_pred)
        if len(unique) < 2:
            return float("nan")
        return float(fn(X_data, labels_pred))
    except Exception as e:
        print(f"[WARN] Failed to compute {name}: {e}")
        return float("nan")


def compute_distance_matrix(X: np.ndarray) -> np.ndarray:
    """
    Full pairwise Euclidean distance matrix for X (n, d).
    """
    # (x - y)^2 = ||x||^2 + ||y||^2 - 2 x·y
    sq = np.sum(X ** 2, axis=1, keepdims=True)  # (n, 1)
    D_sq = sq + sq.T - 2.0 * (X @ X.T)
    np.maximum(D_sq, 0.0, out=D_sq)
    D = np.sqrt(D_sq)
    return D


def compute_knn(
    D: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    For distance matrix D (n, n), compute:
      - knn_indices[i]: indices of k nearest neighbors of i (excluding i itself).
      - knn_sets[i]: the same as a numpy array, but pre-wrapped for easy intersection.
    """
    n = D.shape[0]
    # argsort per row, then skip self at position 0
    order = np.argsort(D, axis=1)
    knn_indices = order[:, 1 : k + 1]
    knn_sets = [knn_indices[i] for i in range(n)]
    return knn_indices, knn_sets


def compute_ins_matrix(
    D: np.ndarray,
    knn_sets: List[np.ndarray],
) -> np.ndarray:
    """
    Compute the interactive neighbors similarity matrix Ins(i,j) for a set of points.

    Ins(i,j) ~ (1 / avg distance among the union of neighbors around i and j) * (# shared neighbors)

    Using the definition from the paper:
      - The denominator: average distance between all pairs (p,q) where
        p in KNN(i), q in KNN(j).
      - The multiplier: number of shared neighbors between i and j.

    We implement this literally as:
        Ins(i,j) = (k^2 / sum_{p in KNN(i), q in KNN(j)} d_{pq}) * |SNN(i,j)|
    with a small epsilon in the denominator for stability.
    """
    n = D.shape[0]
    k = knn_sets[0].shape[0] if n > 0 else 0
    eps = 1e-12

    Ins = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        K_i = knn_sets[i]
        set_i = set(K_i.tolist())
        for j in range(i + 1, n):
            K_j = knn_sets[j]
            set_j = set(K_j.tolist())

            # Shared nearest neighbors
            snn = set_i.intersection(set_j)
            if not snn:
                continue  # Ins(i,j) remains 0

            # Sum distances between all pairs (p in KNN(i), q in KNN(j))
            block = D[np.ix_(K_i, K_j)]
            sum_dpq = float(block.sum())

            # reciprocal of average distance (k^2 / sum) times |SNN|
            Ins_ij = (k * k / (sum_dpq + eps)) * len(snn)
            Ins[i, j] = Ins_ij
            Ins[j, i] = Ins_ij  # symmetry

    return Ins


def insdpc_on_train(
    X_train: np.ndarray,
    k_neighbors: int,
    k_target: int,
) -> np.ndarray:
    """
    Run INSDPC on the TRAIN set only, returning cluster labels for train points.

    Steps:
      - Compute distance matrix D_train.
      - KNN sets per point.
      - Ins matrix (interactive neighbors similarity).
      - Local density rho_i = sum_j Ins(i,j).
      - Relative distance delta_i (DPC-style, using rho).
      - Decision value gamma_i = rho_i * delta_i.
      - Choose k_target centers with largest gamma_i.
      - Two-step assignment:
          1) Core points via BFS and Ins-based threshold.
          2) Remaining points via neighbor voting (increasing radius if needed).
    """
    n_train = X_train.shape[0]
    if n_train == 0:
        return np.array([], dtype=int)
    if k_neighbors <= 0:
        raise ValueError("k_neighbors must be positive for INSDPC.")

    # --- Distance matrix + KNN ---
    D_train = compute_distance_matrix(X_train)
    k_eff = min(k_neighbors, max(1, n_train - 1))
    knn_indices, knn_sets = compute_knn(D_train, k_eff)

    # --- Ins matrix ---
    Ins = compute_ins_matrix(D_train, knn_sets)

    # --- Local density & relative distance ---
    rho = Ins.sum(axis=1)  # (n_train,)

    # DPC-style relative distance w.r.t. rho
    # sort indices by rho descending
    order = np.argsort(-rho)
    delta = np.zeros_like(rho)
    # the one with max rho: distance to farthest point
    max_rho_idx = order[0]
    delta[max_rho_idx] = D_train[max_rho_idx].max()
    # for others: distance to nearest point with higher rho
    for idx in order[1:]:
        higher = order[order < idx]  # not correct; we want "indices with higher rho", i.e. previously processed
        # Better: 'higher' = all indices in 'order' up to the one whose position is before idx's position.

    # Correct computation of delta:
    # We'll recompute properly using a more explicit loop.
    delta = np.zeros_like(rho)
    max_rho_idx = order[0]
    delta[max_rho_idx] = D_train[max_rho_idx].max()
    processed = [max_rho_idx]
    for idx in order[1:]:
        # min distance to any processed (higher rho) point
        dists_to_higher = D_train[idx, processed]
        delta[idx] = float(dists_to_higher.min())
        processed.append(idx)

    # --- Decision value gamma and center selection ---
    gamma = rho * delta
    # choose k_target centers with largest gamma
    if k_target > n_train:
        k_target = n_train
    center_indices = np.argsort(-gamma)[:k_target]
    # labels: -1 = unassigned; 0..k_target-1 = cluster ids
    labels = -1 * np.ones(n_train, dtype=int)
    for cid, idx in enumerate(center_indices):
        labels[idx] = cid

    # --- Two-step assignment ---

    # 1) Core points assignment via BFS over Ins threshold
    from collections import deque
    Q = deque(center_indices.tolist())

    while Q:
        i = Q.popleft()
        ci = labels[i]
        if ci < 0:
            continue

        neighbors = knn_indices[i]
        # average interactive neighbors similarity to its knn
        ins_vals = Ins[i, neighbors]
        if len(ins_vals) == 0:
            continue
        mean_ins = float(ins_vals.mean())

        for j in neighbors:
            if labels[j] != -1:
                continue  # already assigned
            if Ins[i, j] > mean_ins:
                labels[j] = ci
                Q.append(j)

    # 2) Remaining points via neighbor voting (increasing radius)
    unassigned = np.where(labels == -1)[0]
    if len(unassigned) > 0:
        n = n_train
        for idx in unassigned:
            # Sort all other points by distance
            order_i = np.argsort(D_train[idx])
            # skip self at index 0
            order_i = order_i[1:]
            assigned_label = -1

            # start from k_eff neighbors, grow the neighborhood until we see assigned labels
            for k_cur in range(k_eff, n):
                neigh = order_i[:k_cur]
                neigh_labels = labels[neigh]
                mask_assigned = neigh_labels != -1
                if np.any(mask_assigned):
                    # majority vote among assigned neighbors
                    vals, counts = np.unique(neigh_labels[mask_assigned], return_counts=True)
                    assigned_label = int(vals[np.argmax(counts)])
                    break

            if assigned_label == -1:
                # fallback: assign to nearest center
                d_to_centers = D_train[idx, center_indices]
                assigned_label = int(np.argmin(d_to_centers))
            labels[idx] = assigned_label

    return labels


# ============================= MAIN ================================

def insdpc_main(args=None) -> Dict[str, Any]:
    if args is None:
        args = parse_args()

    dataset = args.dataset.lower()
    set_seed(args.seed)

    data_root = Path(args.data_root)
    root_out_dir = Path(args.out_dir)
    out_dir = root_out_dir / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Outputs will be saved under: {out_dir}")
    print(f"[INFO] Figures will be saved under: {figures_dir}")

    # Load dataset (full)
    X, y = load_dataset(dataset, base_dir=str(data_root))
    print(f"[INFO] Loaded dataset '{dataset}': X.shape={X.shape}, unique labels={np.unique(y)}")

    # Train/test split
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split_with_strategy(
        X, y,
        train_frac=args.train_fraction,
        strategy=args.split_strategy,
        seed=args.seed,
    )
    print(f"[INFO] Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Run INSDPC on TRAIN
    print(f"[INFO] Running INSDPC on train set (k_neighbors={args.k_neighbors}, k_target={args.k_target}) ...")
    y_pred_train = insdpc_on_train(X_train, k_neighbors=args.k_neighbors, k_target=args.k_target)

    # Assign TEST labels via 1-NN from train
    if len(X_train) > 0:
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(X_train)
        dists_test, idx_test_nn = nn.kneighbors(X_test)
        idx_test_nn = idx_test_nn.squeeze(1)
        y_pred_test = y_pred_train[idx_test_nn]
    else:
        y_pred_test = -1 * np.ones(len(X_test), dtype=int)

    # ==================== METRICS =================================

    # External metrics
    nmi_train = normalized_mutual_info_score(y_train, y_pred_train)
    nmi_test = normalized_mutual_info_score(y_test, y_pred_test)

    ami_train = adjusted_mutual_info_score(y_train, y_pred_train)
    ami_test = adjusted_mutual_info_score(y_test, y_pred_test)

    ari_train = adjusted_rand_score(y_train, y_pred_train)
    ari_test = adjusted_rand_score(y_test, y_pred_test)

    fmi_train = fowlkes_mallows_score(y_train, y_pred_train)
    fmi_test = fowlkes_mallows_score(y_test, y_pred_test)

    # Accuracy with permutation
    acc_train, map_train = clustering_accuracy_with_map(y_train, y_pred_train)
    acc_test, map_test = clustering_accuracy_with_map(y_test, y_pred_test)

    # Internal metrics
    sil_train = safe_cluster_metric(silhouette_score, X_train, y_pred_train, "silhouette (train)")
    sil_test = safe_cluster_metric(silhouette_score, X_test, y_pred_test, "silhouette (test)")

    db_train_val = safe_cluster_metric(davies_bouldin_score, X_train, y_pred_train, "Davies-Bouldin (train)")
    db_test_val = safe_cluster_metric(davies_bouldin_score, X_test, y_pred_test, "Davies-Bouldin (test)")

    ch_train = safe_cluster_metric(calinski_harabasz_score, X_train, y_pred_train, "Calinski-Harabasz (train)")
    ch_test = safe_cluster_metric(calinski_harabasz_score, X_test, y_pred_test, "Calinski-Harabasz (test)")

    print(f"[RESULT] Dataset = {dataset} (INSDPC)")
    print(f"[RESULT] Train NMI   = {nmi_train:.4f}, ACC = {acc_train:.4f}")
    print(f"[RESULT] Train AMI   = {ami_train:.4f}, ARI = {ari_train:.4f}, FMI = {fmi_train:.4f}")
    print(f"[RESULT] Train Sil   = {sil_train:.4f}, DB  = {db_train_val:.4f}, CH  = {ch_train:.4f}")
    print(f"[RESULT] Test  NMI   = {nmi_test:.4f}, ACC = {acc_test:.4f}")
    print(f"[RESULT] Test  AMI   = {ami_test:.4f}, ARI = {ari_test:.4f}, FMI = {fmi_test:.4f}")
    print(f"[RESULT] Test  Sil   = {sil_test:.4f}, DB  = {db_test_val:.4f}, CH  = {ch_test:.4f}")

    # Map test predictions for visualization
    y_pred_test_mapped = apply_label_mapping(y_pred_test, map_test)
    correct_test = (y_pred_test_mapped == y_test)

    # ==================== PLOTS: TRAIN + TEST ======================

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 2D coordinates for scatter plots
    if X_train.shape[1] >= 2:
        train_x = X_train[:, 0]
        train_y = X_train[:, 1]
        test_x = X_test[:, 0]
        test_y = X_test[:, 1]
    elif X_train.shape[1] == 1:
        train_x = X_train[:, 0]
        train_y = np.zeros_like(train_x)
        test_x = X_test[:, 0]
        test_y = np.zeros_like(test_x)
    else:
        raise ValueError("X must have at least 1 feature.")

    # (a) Train ground truth
    ax = axes[0]
    ax.scatter(
        train_x,
        train_y,
        c=y_train,
        cmap="tab10",
        s=20,
        alpha=0.9,
    )
    ax.set_title("Train ground truth")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # (b) Train INSDPC clusters (title: NMI & ACC only)
    ax = axes[1]
    ax.scatter(
        train_x,
        train_y,
        c=y_pred_train,
        cmap="tab10",
        s=20,
        alpha=0.9,
    )
    ax.set_title(f"Train INSDPC (NMI={nmi_train:.3f}, ACC={acc_train:.3f})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # (c) Test: base points + correct + mis-clustered
    ax = axes[2]
    ax.scatter(
        test_x,
        test_y,
        c="blue",
        s=20,
        alpha=0.3,
        label="Test points",
    )
    ax.scatter(
        test_x[correct_test],
        test_y[correct_test],
        c="green",
        s=20,
        alpha=0.9,
        label="Correct",
    )
    ax.scatter(
        test_x[~correct_test],
        test_y[~correct_test],
        c="red",
        s=30,
        alpha=0.9,
        marker="x",
        label="Mis-clustered",
    )
    ax.set_title(f"Test INSDPC (NMI={nmi_test:.3f}, ACC={acc_test:.3f})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    train_test_path = figures_dir / f"{dataset}_insdpc_train_test.png"
    fig.savefig(train_test_path, dpi=150, transparent=True)
    print(f"[INFO] Saved INSDPC train/test figure to: {train_test_path}")
    if args.show_plots:
        fig.show()
    else:
        plt.close(fig)

    # ========== EXTRA PLOT: FULL-DATASET GROUND TRUTH =============

    try:
        if X.shape[1] >= 2:
            full_x = X[:, 0]
            full_y = X[:, 1]
        elif X.shape[1] == 1:
            full_x = X[:, 0]
            full_y = np.zeros_like(full_x)
        else:
            raise ValueError("X must have at least 1 feature for plotting.")

        fig_full, ax_full = plt.subplots(figsize=(6, 5))
        ax_full.scatter(
            full_x,
            full_y,
            c=y,
            cmap="tab10",
            s=15,
            alpha=0.9,
        )
        ax_full.set_title("Full dataset ground truth")
        ax_full.set_xlabel("x")
        ax_full.set_ylabel("y")
        fig_full.tight_layout()

        full_gt_path = figures_dir / f"{dataset}_insdpc_full_ground_truth.png"
        fig_full.savefig(full_gt_path, dpi=150, transparent=True)
        print(f"[INFO] Saved full dataset GT figure (INSDPC run) to: {full_gt_path}")
        if args.show_plots:
            fig_full.show()
        else:
            plt.close(fig_full)
    except Exception as e:
        print(f"[WARN] Failed to save full dataset ground-truth plot: {e}")

    # ================ METRIC BAR PLOTS (individual) ==============
    try:
        metric_values = {
            "NMI": (nmi_train, nmi_test),
            "AMI": (ami_train, ami_test),
            "ARI": (ari_train, ari_test),
            "FMI": (fmi_train, fmi_test),
            "Silhouette": (sil_train, sil_test),
            "DaviesBouldin": (db_train_val, db_test_val),
            "CalinskiHarabasz": (ch_train, ch_test),
            "ACC": (acc_train, acc_test),
        }

        for name, (v_train, v_test) in metric_values.items():
            fig_m, ax_m = plt.subplots(figsize=(4, 4))
            ax_m.bar(["Train", "Test"], [v_train, v_test])
            ax_m.set_title(f"{name} (INSDPC, train vs test)")
            ax_m.set_ylabel(name)
            fig_m.tight_layout()
            metric_path = figures_dir / f"{dataset}_insdpc_metric_{name.lower()}.png"
            fig_m.savefig(metric_path, dpi=150, transparent=True)
            print(f"[INFO] Saved INSDPC metric plot '{name}' to: {metric_path}")
            if args.show_plots:
                fig_m.show()
            else:
                plt.close(fig_m)
    except Exception as e:
        print(f"[WARN] Failed to save INSDPC metric bar plots: {e}")

    # ================ SAVE TEXT RESULTS / METRICS ==================

    results_txt_path = out_dir / f"{dataset}_insdpc_results.txt"
    with open(results_txt_path, "w") as f:
        f.write(f"dataset          = {dataset}\n")
        f.write(f"run_id           = {args.run_id}\n")
        f.write(f"seed             = {args.seed}\n")
        f.write(f"split_strategy   = {args.split_strategy}\n")
        f.write(f"train_fraction   = {args.train_fraction}\n")
        f.write(f"k_neighbors      = {args.k_neighbors}\n")
        f.write(f"k_target         = {args.k_target}\n")
        f.write("\n")
        f.write(f"Train NMI        = {nmi_train:.6f}\n")
        f.write(f"Train AMI        = {ami_train:.6f}\n")
        f.write(f"Train ARI        = {ari_train:.6f}\n")
        f.write(f"Train FMI        = {fmi_train:.6f}\n")
        f.write(f"Train Silhouette = {sil_train:.6f}\n")
        f.write(f"Train DB         = {db_train_val:.6f}\n")
        f.write(f"Train CH         = {ch_train:.6f}\n")
        f.write(f"Train ACC        = {acc_train:.6f}\n")
        f.write("\n")
        f.write(f"Test NMI         = {nmi_test:.6f}\n")
        f.write(f"Test AMI         = {ami_test:.6f}\n")
        f.write(f"Test ARI         = {ari_test:.6f}\n")
        f.write(f"Test FMI         = {fmi_test:.6f}\n")
        f.write(f"Test Silhouette  = {sil_test:.6f}\n")
        f.write(f"Test DB          = {db_test_val:.6f}\n")
        f.write(f"Test CH          = {ch_test:.6f}\n")
        f.write(f"Test ACC         = {acc_test:.6f}\n")

    print(f"[INFO] Saved INSDPC scalar results to: {results_txt_path}")

    # ================== APPEND RUN SUMMARY TO CSV ==================
    summary_csv_path = Path.cwd() / "results_summary.csv"

    row = {
        # identifiers
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "dataset": dataset,
        "model": MODEL_NAME,
        "run_id": args.run_id,

        # configs
        "seed": args.seed,
        "split_strategy": args.split_strategy,
        "train_fraction": args.train_fraction,

        "k_neighbors": args.k_neighbors,
        "k_target": args.k_target,

        # metrics (train)
        "train_nmi": nmi_train,
        "train_ami": ami_train,
        "train_ari": ari_train,
        "train_fmi": fmi_train,
        "train_sil": sil_train,
        "train_db": db_train_val,
        "train_ch": ch_train,
        "train_acc": acc_train,

        # metrics (test)
        "test_nmi": nmi_test,
        "test_ami": ami_test,
        "test_ari": ari_test,
        "test_fmi": fmi_test,
        "test_sil": sil_test,
        "test_db": db_test_val,
        "test_ch": ch_test,
        "test_acc": acc_test,

        # artifacts
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
        "train_test_png": str(train_test_path),
        "figures_dir": str(figures_dir),
        "summary_csv": str(summary_csv_path),
    }


if __name__ == "__main__":
    insdpc_main()
