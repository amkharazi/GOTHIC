"""
mdbscan_model.py

MDBSCAN baseline (Multi-level DBSCAN):
  - Scan a range of eps values.
  - For each eps, run DBSCAN on TRAIN set.
  - Choose the eps with the best silhouette score (unsupervised).
  - Assign TEST labels via 1-NN from train.
  - Compute external + internal metrics and save plots.

Matches the style of:
  gothic_model.py, kmeans_model.py, dbscan_model.py, hdbscan_model.py, insdpc_model.py, amd_dbscan_model.py
"""

import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
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

DEFAULT_EPS_MIN = 0.05
DEFAULT_EPS_MAX = 1.0
DEFAULT_N_LEVELS = 10
DEFAULT_MIN_SAMPLES = 5


# =========================== ARG PARSER ============================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="MDBSCAN (multi-level DBSCAN) baseline for clustering.")

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
        default="mdbscan_default",
        help="Run identifier; outputs go under out_dir/run_id/.",
    )

    # MDBSCAN params
    parser.add_argument(
        "--eps_min",
        type=float,
        default=DEFAULT_EPS_MIN,
        help="Minimum eps value to scan.",
    )
    parser.add_argument(
        "--eps_max",
        type=float,
        default=DEFAULT_EPS_MAX,
        help="Maximum eps value to scan.",
    )
    parser.add_argument(
        "--n_levels",
        type=int,
        default=DEFAULT_N_LEVELS,
        help="Number of eps levels between eps_min and eps_max.",
    )
    parser.add_argument(
        "--min_samples",
        type=int,
        default=DEFAULT_MIN_SAMPLES,
        help="min_samples parameter for DBSCAN.",
    )

    # Misc
    parser.add_argument(
        "--show_plots",
        action="store_true",
        help="If set, show matplotlib figures interactively after saving.",
    )

    return parser.parse_args(argv)


# =========================== UTILITIES =============================

def train_test_split_with_strategy(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float = 0.8,
    strategy: str = "balanced",
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Same behavior as in the other baselines:
      - 'random'   : simple random split.
      - 'balanced' : per-class, density-aware split.
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

        from sklearn.neighbors import NearestNeighbors

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

            # Density estimate via k-NN within class
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
        train_idx, test_idx,
    )


def safe_cluster_metric(fn, X_data, labels_pred, name: str) -> float:
    """
    Compute an internal clustering metric, but handle edge cases gracefully.
    """
    try:
        if len(np.unique(labels_pred)) < 2:
            return float("nan")
        return float(fn(X_data, labels_pred))
    except Exception as e:
        print(f"[WARN] Failed to compute {name}: {e}")
        return float("nan")


def mdbscan_cluster(
    X_train: np.ndarray,
    eps_min: float,
    eps_max: float,
    n_levels: int,
    min_samples: int,
) -> Tuple[np.ndarray, float]:
    """
    Run multi-level DBSCAN on X_train:

      - eps values linearly spaced between eps_min and eps_max (inclusive).
      - For each eps:
          * run DBSCAN(eps, min_samples).
          * compute silhouette if >= 2 clusters.
      - pick eps with the best silhouette.
      - return labels for that eps and the chosen eps.

    Completely unsupervised (no labels used).
    """
    n = X_train.shape[0]
    if n == 0:
        return np.array([], dtype=int), eps_min

    if n_levels <= 1:
        eps_values = np.array([eps_min])
    else:
        eps_values = np.linspace(eps_min, eps_max, n_levels)

    best_sil = -np.inf
    best_labels = None
    best_eps = eps_values[0]

    for eps in eps_values:
        db = DBSCAN(
            eps=float(eps),
            min_samples=min_samples,
            metric="euclidean",
        )
        labels = db.fit_predict(X_train)

        # We want at least 2 clusters for silhouette
        unique = np.unique(labels)
        if len(unique) < 2:
            sil = -np.inf
        else:
            try:
                sil = float(silhouette_score(X_train, labels))
            except Exception as e:
                print(f"[WARN] silhouette failed for eps={eps:.4f}: {e}")
                sil = -np.inf

        print(f"[INFO] MDBSCAN level: eps={eps:.4f}, clusters={len(unique)}, silhouette={sil:.4f}")
        if sil > best_sil:
            best_sil = sil
            best_labels = labels
            best_eps = float(eps)

    if best_labels is None:
        # fallback: everything noise
        best_labels = -1 * np.ones(n, dtype=int)

    print(f"[INFO] MDBSCAN chose eps={best_eps:.4f} with silhouette={best_sil:.4f}")
    return best_labels, best_eps


# ============================= MAIN ================================

def mdbscan_main(args=None) -> Dict[str, Any]:
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

    # Load dataset
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

    # Run MDBSCAN on TRAIN
    print(
        f"[INFO] Running MDBSCAN on train set (eps_min={args.eps_min}, "
        f"eps_max={args.eps_max}, n_levels={args.n_levels}, min_samples={args.min_samples}) ..."
    )
    y_pred_train, chosen_eps = mdbscan_cluster(
        X_train,
        eps_min=args.eps_min,
        eps_max=args.eps_max,
        n_levels=args.n_levels,
        min_samples=args.min_samples,
    )

    # Assign TEST labels via 1-NN from train
    if len(X_train) > 0 and len(X_test) > 0:
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

    print(f"[RESULT] Dataset = {dataset} (MDBSCAN)")
    print(f"[RESULT] Chosen eps = {chosen_eps:.4f}")
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

    # (b) Train MDBSCAN clusters
    ax = axes[1]
    ax.scatter(
        train_x,
        train_y,
        c=y_pred_train,
        cmap="tab10",
        s=20,
        alpha=0.9,
    )
    ax.set_title(f"Train MDBSCAN (eps={chosen_eps:.3f}, NMI={nmi_train:.3f}, ACC={acc_train:.3f})")
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
    ax.set_title(f"Test MDBSCAN (NMI={nmi_test:.3f}, ACC={acc_test:.3f})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    train_test_path = figures_dir / f"{dataset}_mdbscan_train_test.png"
    fig.savefig(train_test_path, dpi=150, transparent=True)
    print(f"[INFO] Saved MDBSCAN train/test figure to: {train_test_path}")
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

        full_gt_path = figures_dir / f"{dataset}_mdbscan_full_ground_truth.png"
        fig_full.savefig(full_gt_path, dpi=150, transparent=True)
        print(f"[INFO] Saved full dataset GT figure (MDBSCAN run) to: {full_gt_path}")
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
            ax_m.set_title(f"{name} (MDBSCAN, train vs test)")
            ax_m.set_ylabel(name)
            fig_m.tight_layout()
            metric_path = figures_dir / f"{dataset}_mdbscan_metric_{name.lower()}.png"
            fig_m.savefig(metric_path, dpi=150, transparent=True)
            print(f"[INFO] Saved MDBSCAN metric plot '{name}' to: {metric_path}")
            if args.show_plots:
                fig_m.show()
            else:
                plt.close(fig_m)
    except Exception as e:
        print(f"[WARN] Failed to save MDBSCAN metric bar plots: {e}")

    # ================ SAVE TEXT RESULTS / METRICS ==================

    results_txt_path = out_dir / f"{dataset}_mdbscan_results.txt"
    with open(results_txt_path, "w") as f:
        f.write(f"dataset          = {dataset}\n")
        f.write(f"run_id           = {args.run_id}\n")
        f.write(f"seed             = {args.seed}\n")
        f.write(f"split_strategy   = {args.split_strategy}\n")
        f.write(f"train_fraction   = {args.train_fraction}\n")
        f.write(f"eps_min          = {args.eps_min:.6f}\n")
        f.write(f"eps_max          = {args.eps_max:.6f}\n")
        f.write(f"n_levels         = {args.n_levels}\n")
        f.write(f"min_samples      = {args.min_samples}\n")
        f.write(f"chosen_eps       = {chosen_eps:.6f}\n")
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

    print(f"[INFO] Saved MDBSCAN scalar results to: {results_txt_path}")

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
    }


if __name__ == "__main__":
    mdbscan_main()
