"""
spectral_model.py

Spectral Clustering baseline with the same data handling & metrics as GOTHIC/KMeans/DBSCAN/HDBSCAN.

Pipeline:
  1) Load dataset via loader.load_dataset.
  2) Train/test split (random or density-balanced).
  3) Fit SpectralClustering on the TRAIN set.
  4) Predict clusters:
       - train: Spectral labels
       - test : nearest-neighbor projection from train labels (1-NN in feature space)
  5) Compute metrics:
       - NMI, AMI, ARI, FMI, ACC
       - Silhouette, Davies–Bouldin, Calinski–Harabasz
  6) Save plots (transparent) and results under out_dir/run_id/.
"""

MODEL_NAME = "spectral"

import csv
import json
from datetime import datetime

import argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import SpectralClustering
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

# Spectral params (defaults are reasonable for small datasets)
DEFAULT_AFFINITY = "rbf"  # "rbf" or "nearest_neighbors"
DEFAULT_GAMMA = 1.0
DEFAULT_N_NEIGHBORS = 10
DEFAULT_ASSIGN_LABELS = "kmeans"  # "kmeans" or "discretize"

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
    "digits": 10,
    "olivetti_faces": 40,
}


# =========================== ARG PARSER ============================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Spectral Clustering baseline for clustering.")

    # Core
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train_fraction", type=float, default=DEFAULT_TRAIN_FRACTION)
    parser.add_argument(
        "--split_strategy",
        type=str,
        default=DEFAULT_SPLIT_STRATEGY,
        choices=["random", "balanced"],
        help="Train/test split strategy.",
    )
    parser.add_argument("--data_root", type=str, default="datasets")
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--run_id", type=str, default="spectral_default")

    # Target clusters
    parser.add_argument(
        "--k_target",
        type=int,
        default=-1,
        help="Target number of clusters; if <= 0, uses dataset-specific default.",
    )

    # Spectral
    parser.add_argument(
        "--affinity",
        type=str,
        default=DEFAULT_AFFINITY,
        choices=["rbf", "nearest_neighbors"],
        help="Affinity type for SpectralClustering.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=DEFAULT_GAMMA,
        help="Gamma for RBF affinity (used when affinity=rbf).",
    )
    parser.add_argument(
        "--n_neighbors",
        type=int,
        default=DEFAULT_N_NEIGHBORS,
        help="Number of neighbors (used when affinity=nearest_neighbors).",
    )
    parser.add_argument(
        "--assign_labels",
        type=str,
        default=DEFAULT_ASSIGN_LABELS,
        choices=["kmeans", "discretize"],
        help="Label assignment strategy in SpectralClustering.",
    )

    # Misc
    parser.add_argument("--show_plots", action="store_true")

    return parser.parse_args(argv)


# =========================== UTILITIES =============================

def _to_csv_value(v):
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, (Path,)):
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
    try:
        if len(np.unique(labels_pred)) < 2:
            return float("nan")
        return float(fn(X_data, labels_pred))
    except Exception as e:
        print(f"[WARN] Failed to compute {name}: {e}")
        return float("nan")


# ============================= MAIN ================================

def spectral_main(args=None) -> Dict[str, Any]:
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

    # Fit SpectralClustering on train (no predict() in sklearn => we do 1-NN projection for test)
    print(
        "[INFO] Fitting SpectralClustering on train set "
        f"(k_target={k_target}, affinity={args.affinity}, assign_labels={args.assign_labels}) ..."
    )

    sc_kwargs = dict(
        n_clusters=k_target,
        affinity=args.affinity,
        assign_labels=args.assign_labels,
        random_state=args.seed,
    )
    if args.affinity == "rbf":
        sc_kwargs["gamma"] = args.gamma
    elif args.affinity == "nearest_neighbors":
        sc_kwargs["n_neighbors"] = max(2, int(args.n_neighbors))

    sc = SpectralClustering(**sc_kwargs)
    y_pred_train = sc.fit_predict(X_train)

    # Test labels via nearest train point label
    if len(X_train) > 0:
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(X_train)
        _, idx_test_nn = nn.kneighbors(X_test)
        idx_test_nn = idx_test_nn.squeeze(1)
        y_pred_test = y_pred_train[idx_test_nn]
    else:
        y_pred_test = np.zeros(len(X_test), dtype=int)

    # ==================== METRICS =================================

    nmi_train = normalized_mutual_info_score(y_train, y_pred_train)
    nmi_test = normalized_mutual_info_score(y_test, y_pred_test)

    ami_train = adjusted_mutual_info_score(y_train, y_pred_train)
    ami_test = adjusted_mutual_info_score(y_test, y_pred_test)

    ari_train = adjusted_rand_score(y_train, y_pred_train)
    ari_test = adjusted_rand_score(y_test, y_pred_test)

    fmi_train = fowlkes_mallows_score(y_train, y_pred_train)
    fmi_test = fowlkes_mallows_score(y_test, y_pred_test)

    acc_train, map_train = clustering_accuracy_with_map(y_train, y_pred_train)
    acc_test, map_test = clustering_accuracy_with_map(y_test, y_pred_test)

    sil_train = safe_cluster_metric(silhouette_score, X_train, y_pred_train, "silhouette (train)")
    sil_test = safe_cluster_metric(silhouette_score, X_test, y_pred_test, "silhouette (test)")

    db_train_val = safe_cluster_metric(davies_bouldin_score, X_train, y_pred_train, "Davies-Bouldin (train)")
    db_test_val = safe_cluster_metric(davies_bouldin_score, X_test, y_pred_test, "Davies-Bouldin (test)")

    ch_train = safe_cluster_metric(calinski_harabasz_score, X_train, y_pred_train, "Calinski-Harabasz (train)")
    ch_test = safe_cluster_metric(calinski_harabasz_score, X_test, y_pred_test, "Calinski-Harabasz (test)")

    print(f"[RESULT] Dataset = {dataset} (Spectral)")
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

    # Use first 2 features for plotting
    if X_train.shape[1] >= 2:
        train_x, train_y2 = X_train[:, 0], X_train[:, 1]
        test_x, test_y2 = X_test[:, 0], X_test[:, 1]
    elif X_train.shape[1] == 1:
        train_x, train_y2 = X_train[:, 0], np.zeros_like(X_train[:, 0])
        test_x, test_y2 = X_test[:, 0], np.zeros_like(X_test[:, 0])
    else:
        raise ValueError("X must have at least 1 feature.")

    # (a) Train ground truth
    ax = axes[0]
    ax.scatter(train_x, train_y2, c=y_train, cmap="tab10", s=20, alpha=0.9)
    ax.set_title("Train ground truth")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # (b) Train predicted
    ax = axes[1]
    ax.scatter(train_x, train_y2, c=y_pred_train, cmap="tab10", s=20, alpha=0.9)
    ax.set_title(f"Train Spectral (NMI={nmi_train:.3f}, ACC={acc_train:.3f})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # (c) Test correctness view
    ax = axes[2]
    ax.scatter(test_x, test_y2, c="blue", s=20, alpha=0.3, label="Test points")
    ax.scatter(test_x[correct_test], test_y2[correct_test], c="green", s=20, alpha=0.9, label="Correct")
    ax.scatter(test_x[~correct_test], test_y2[~correct_test], c="red", s=30, alpha=0.9, marker="x", label="Mis-clustered")
    ax.set_title(f"Test Spectral (NMI={nmi_test:.3f}, ACC={acc_test:.3f})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    train_test_path = figures_dir / f"{dataset}_spectral_train_test.png"
    fig.savefig(train_test_path, dpi=150, transparent=True)
    print(f"[INFO] Saved Spectral train/test figure to: {train_test_path}")
    if args.show_plots:
        fig.show()
    else:
        plt.close(fig)

    # Full dataset GT plot
    try:
        if X.shape[1] >= 2:
            full_x, full_y2 = X[:, 0], X[:, 1]
        elif X.shape[1] == 1:
            full_x, full_y2 = X[:, 0], np.zeros_like(X[:, 0])
        else:
            raise ValueError("X must have at least 1 feature for plotting.")

        fig_full, ax_full = plt.subplots(figsize=(6, 5))
        ax_full.scatter(full_x, full_y2, c=y, cmap="tab10", s=15, alpha=0.9)
        ax_full.set_title("Full dataset ground truth")
        ax_full.set_xlabel("x")
        ax_full.set_ylabel("y")
        fig_full.tight_layout()

        full_gt_path = figures_dir / f"{dataset}_spectral_full_ground_truth.png"
        fig_full.savefig(full_gt_path, dpi=150, transparent=True)
        print(f"[INFO] Saved full dataset GT figure (Spectral run) to: {full_gt_path}")
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
            "DaviesBouldin": (db_train_val, db_test_val),
            "CalinskiHarabasz": (ch_train, ch_test),
            "ACC": (acc_train, acc_test),
        }

        for name_m, (v_train, v_test) in metric_values.items():
            fig_m, ax_m = plt.subplots(figsize=(4, 4))
            ax_m.bar(["Train", "Test"], [v_train, v_test])
            ax_m.set_title(f"{name_m} (Spectral, train vs test)")
            ax_m.set_ylabel(name_m)
            fig_m.tight_layout()
            metric_path = figures_dir / f"{dataset}_spectral_metric_{name_m.lower()}.png"
            fig_m.savefig(metric_path, dpi=150, transparent=True)
            print(f"[INFO] Saved Spectral metric plot '{name_m}' to: {metric_path}")
            if args.show_plots:
                fig_m.show()
            else:
                plt.close(fig_m)
    except Exception as e:
        print(f"[WARN] Failed to save Spectral metric bar plots: {e}")

    # Save scalar results
    results_txt_path = out_dir / f"{dataset}_spectral_results.txt"
    with open(results_txt_path, "w") as f:
        f.write(f"dataset          = {dataset}\n")
        f.write(f"run_id           = {args.run_id}\n")
        f.write(f"seed             = {args.seed}\n")
        f.write(f"split_strategy   = {args.split_strategy}\n")
        f.write(f"train_fraction   = {args.train_fraction}\n")
        f.write(f"k_target         = {k_target}\n")
        f.write(f"affinity         = {args.affinity}\n")
        f.write(f"gamma            = {args.gamma}\n")
        f.write(f"n_neighbors      = {args.n_neighbors}\n")
        f.write(f"assign_labels    = {args.assign_labels}\n")
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

    print(f"[INFO] Saved Spectral scalar results to: {results_txt_path}")

    # Append run summary
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
        "affinity": args.affinity,
        "gamma": args.gamma,
        "n_neighbors": args.n_neighbors,
        "assign_labels": args.assign_labels,
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
        "train_test_png": str(train_test_path),
        "figures_dir": str(figures_dir),
        "summary_csv": str(summary_csv_path),
    }


if __name__ == "__main__":
    spectral_main()
