"""
kmeans_model.py

Simple KMeans baseline with the same data handling & metrics
as GOTHIC, so we can compare them fairly.

Pipeline:
  1) Load dataset via loader.load_dataset.
  2) Train/test split (random or density-balanced).
  3) Fit KMeans with K = k_target on train.
  4) Predict clusters for train & test.
  5) Compute metrics:
       - NMI, AMI, ARI, FMI, ACC
       - Silhouette, Davies–Bouldin, Calinski–Harabasz
  6) Save plots (transparent) and results under out_dir/run_id/.
"""

import argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt

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
DEFAULT_KMEANS_N_INIT = 50
DEFAULT_KMEANS_MAX_ITER = 500

# Dataset-specific default number of clusters (if user does not specify --k_target)
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
}


# =========================== ARG PARSER ============================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Simple KMeans baseline for clustering.")

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
        default="kmeans_default",
        help="Run identifier; outputs go under out_dir/run_id/.",
    )

    # KMeans
    parser.add_argument(
        "--kmeans_n_init",
        type=int,
        default=DEFAULT_KMEANS_N_INIT,
        help="n_init parameter for KMeans.",
    )
    parser.add_argument(
        "--kmeans_max_iter",
        type=int,
        default=DEFAULT_KMEANS_MAX_ITER,
        help="max_iter parameter for KMeans.",
    )

    # target clusters
    parser.add_argument(
        "--k_target",
        type=int,
        default=-1,
        help="Target number of clusters; if <= 0, uses dataset-specific default.",
    )

    # Misc
    parser.add_argument(
        "--show_plots",
        action="store_true",
        help="If set, show matplotlib figures interactively after saving.",
    )

    return parser.parse_args(argv)


# =========================== UTILITIES =============================

def train_test_split_with_strategy(X, y, train_frac=0.8, strategy="balanced", seed=42):
    """
    Same behavior as in gothic_model.py:
      - 'random'   : simple random split.
      - 'balanced' : per-cluster density-aware split where denser points
                     are more likely to go to test, and more boundary points
                     stay in train.
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

            # k-NN within cluster for density estimation
            k_nn = min(10, n_c - 1)
            nn = NearestNeighbors(n_neighbors=k_nn + 1)
            nn.fit(X_c)
            dists, _ = nn.kneighbors(X_c)
            # ignore self distance
            mean_dists = dists[:, 1:].mean(axis=1)
            densities = 1.0 / (eps + mean_dists)

            # densest → interior → send more to test
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


# ============================= MAIN ================================

def kmeans_main(args=None) -> Dict[str, Any]:
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

    # Determine k_target
    if args.k_target is None or args.k_target <= 0:
        k_target = DEFAULT_K_TARGETS.get(dataset, 6)
        print(f"[INFO] Using dataset-specific k_target={k_target} for dataset={dataset}")
    else:
        k_target = args.k_target
        print(f"[INFO] Using user-specified k_target={k_target}")

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

    # Fit KMeans on train
    print(f"[INFO] Fitting KMeans with K={k_target} on train set ...")
    km = KMeans(
        n_clusters=k_target,
        random_state=args.seed,
        n_init=args.kmeans_n_init,
        max_iter=args.kmeans_max_iter,
    )
    km.fit(X_train)

    # Predictions
    y_pred_train = km.labels_
    y_pred_test = km.predict(X_test)

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

    db_train = safe_cluster_metric(davies_bouldin_score, X_train, y_pred_train, "Davies-Bouldin (train)")
    db_test = safe_cluster_metric(davies_bouldin_score, X_test, y_pred_test, "Davies-Bouldin (test)")

    ch_train = safe_cluster_metric(calinski_harabasz_score, X_train, y_pred_train, "Calinski-Harabasz (train)")
    ch_test = safe_cluster_metric(calinski_harabasz_score, X_test, y_pred_test, "Calinski-Harabasz (test)")

    print(f"[RESULT] Dataset = {dataset} (KMeans)")
    print(f"[RESULT] Train NMI   = {nmi_train:.4f}, ACC = {acc_train:.4f}")
    print(f"[RESULT] Train AMI   = {ami_train:.4f}, ARI = {ari_train:.4f}, FMI = {fmi_train:.4f}")
    print(f"[RESULT] Train Sil   = {sil_train:.4f}, DB  = {db_train:.4f}, CH  = {ch_train:.4f}")
    print(f"[RESULT] Test  NMI   = {nmi_test:.4f}, ACC = {acc_test:.4f}")
    print(f"[RESULT] Test  AMI   = {ami_test:.4f}, ARI = {ari_test:.4f}, FMI = {fmi_test:.4f}")
    print(f"[RESULT] Test  Sil   = {sil_test:.4f}, DB  = {db_test:.4f}, CH  = {ch_test:.4f}")

    # Map test predictions to best-aligned labels for visualization
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
    sc0 = ax.scatter(
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

    # (b) Train predicted clusters (title: NMI & ACC only)
    ax = axes[1]
    sc1 = ax.scatter(
        train_x,
        train_y,
        c=y_pred_train,
        cmap="tab10",
        s=20,
        alpha=0.9,
    )
    ax.set_title(f"Train KMeans (NMI={nmi_train:.3f}, ACC={acc_train:.3f})")
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
    ax.set_title(f"Test KMeans (NMI={nmi_test:.3f}, ACC={acc_test:.3f})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    train_test_path = figures_dir / f"{dataset}_kmeans_train_test.png"
    fig.savefig(train_test_path, dpi=150, transparent=True)
    print(f"[INFO] Saved KMeans train/test figure to: {train_test_path}")
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

        full_gt_path = figures_dir / f"{dataset}_kmeans_full_ground_truth.png"
        fig_full.savefig(full_gt_path, dpi=150, transparent=True)
        print(f"[INFO] Saved full dataset GT figure (KMeans run) to: {full_gt_path}")
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
            "DaviesBouldin": (db_train, db_test),
            "CalinskiHarabasz": (ch_train, ch_test),
            "ACC": (acc_train, acc_test),
        }

        for name, (v_train, v_test) in metric_values.items():
            fig_m, ax_m = plt.subplots(figsize=(4, 4))
            ax_m.bar(["Train", "Test"], [v_train, v_test])
            ax_m.set_title(f"{name} (KMeans, train vs test)")
            ax_m.set_ylabel(name)
            fig_m.tight_layout()
            metric_path = figures_dir / f"{dataset}_kmeans_metric_{name.lower()}.png"
            fig_m.savefig(metric_path, dpi=150, transparent=True)
            print(f"[INFO] Saved KMeans metric plot '{name}' to: {metric_path}")
            if args.show_plots:
                fig_m.show()
            else:
                plt.close(fig_m)
    except Exception as e:
        print(f"[WARN] Failed to save KMeans metric bar plots: {e}")

    # ================ SAVE TEXT RESULTS / METRICS ==================

    results_txt_path = out_dir / f"{dataset}_kmeans_results.txt"
    with open(results_txt_path, "w") as f:
        f.write(f"dataset          = {dataset}\n")
        f.write(f"run_id           = {args.run_id}\n")
        f.write(f"seed             = {args.seed}\n")
        f.write(f"split_strategy   = {args.split_strategy}\n")
        f.write(f"train_fraction   = {args.train_fraction}\n")
        f.write(f"k_target         = {k_target}\n")
        f.write(f"kmeans_n_init    = {args.kmeans_n_init}\n")
        f.write(f"kmeans_max_iter  = {args.kmeans_max_iter}\n")
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

    print(f"[INFO] Saved KMeans scalar results to: {results_txt_path}")

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
    kmeans_main()
