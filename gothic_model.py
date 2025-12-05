"""
gothic_model.py

Core implementation of GOTHIC:
  Graph-Overclustered Transformer-based Hierarchical Integrated Clustering.

Pipeline (for a chosen dataset):
  1) Ensure dataset is downloaded (via loader/dataset).
  2) Train/test split (80/20, random or density-balanced).
  3) Over-cluster train set via KMeans into micro-clusters.
  4) Build micro-cluster graph (distance quantile) + pairwise features.
  5) Train Transformer-based pair scorer on the micro-cluster graph (supervised).
  6) Greedy merging via learned scores until K_TARGET.
  7) Label train & test, compute metrics (NMI, ACC, AMI, ARI, FMI, Sil, DB, CH).
  8) Save plots, metrics, and checkpoint to disk under outputs/<run_id>/.
"""

import argparse
import math
import sys
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
    pairwise_distances,
)
from sklearn.neighbors import NearestNeighbors

import torch
import torch.nn as nn

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
DEFAULT_N_TRANSFORMER_LAYERS = 2   # full Transformer depth
DEFAULT_TRAIN_EPOCHS = 2000
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_PRINT_EVERY = 200

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
    parser = argparse.ArgumentParser(description="GOTHIC clustering: graph + transformer-based micro-cluster merging.")

    # Core
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET,
                        help="Dataset name (synthetic or real) as defined in dataset.py")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train_fraction", type=float, default=DEFAULT_TRAIN_FRACTION)
    parser.add_argument(
        "--split_strategy",
        type=str,
        default=DEFAULT_SPLIT_STRATEGY,
        choices=["random", "balanced"],
        help="Train/test split strategy."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="datasets",
        help="Base directory where datasets are stored."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs",
        help="Base directory to save checkpoints, plots, and results."
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default="default",
        help="Run identifier; all outputs go under out_dir/run_id/"
    )

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

    # Misc
    parser.add_argument(
        "--show_plots",
        action="store_true",
        help="If set, show matplotlib figures interactively after saving."
    )

    return parser.parse_args(argv)


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
        # Density-aware balanced split per cluster:
        #  - For each cluster, keep roughly train_frac in train, rest test.
        #  - Use local density estimates to send densest interior points to test,
        #    keeping more boundary points in train.
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

            # k-NN within this cluster to estimate local density
            k_nn = min(10, n_c - 1)
            nn = NearestNeighbors(n_neighbors=k_nn + 1)
            nn.fit(X_c)
            dists, _ = nn.kneighbors(X_c)  # (n_c, k_nn+1), first column is self (0)
            mean_dists = dists[:, 1:].mean(axis=1)   # ignore self

            densities = 1.0 / (eps + mean_dists)

            # Densest points -> more interior; send to test
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


def build_micro_clusters(
    X_train,
    y_train,
    n_micro: int,
    seed: int,
    kmeans_n_init: int,
    kmeans_max_iter: int,
    b_boundary: int,
):
    """
    Over-cluster with KMeans on train only; build micro features (with density & boundary).
    For general dimension d, we store centroid (d dims) + 4 scalar features as node features:
      [log(1+size), radius_mean, radius_std, log_density].
    """
    print(f"[INFO] Over-clustering train set into {n_micro} micro-clusters ...")
    km = KMeans(
        n_clusters=n_micro,
        random_state=seed,
        n_init=kmeans_n_init,
        max_iter=kmeans_max_iter,
    )
    km.fit(X_train)
    train_micro_orig = km.labels_      # size = len(X_train)

    micro_features = []
    micro_major_labels = []
    orig_to_new = {}
    new_to_orig = []
    micro_member_indices = []
    micro_border_indices = []

    eps = 1e-6
    d = X_train.shape[1]

    for m in range(n_micro):
        idx_m = np.where(train_micro_orig == m)[0]
        if len(idx_m) == 0:
            continue
        pts = X_train[idx_m]
        centroid = pts.mean(axis=0)  # [d]
        radii = np.linalg.norm(pts - centroid, axis=1)
        radius_mean = float(radii.mean())
        radius_std = float(radii.std()) if len(radii) > 1 else 0.0
        size = len(pts)
        mj = majority_label(y_train[idx_m])

        # Approximate log density in d-D using "radius_mean" as scale
        volume_proxy = (radius_mean ** d) + eps
        density = size / volume_proxy
        log_density = math.log(1.0 + density)

        scalar_feat = np.array([
            math.log(1.0 + size),
            radius_mean,
            radius_std,
            log_density,
        ], dtype=np.float32)

        feat = np.concatenate([centroid.astype(np.float32), scalar_feat], axis=0)

        # Boundary points: farthest from centroid
        order_r = np.argsort(-radii)
        boundary_local = order_r[:b_boundary]
        boundary_global_idx = idx_m[boundary_local]  # indices into X_train

        new_id = len(micro_features)
        orig_to_new[m] = new_id
        new_to_orig.append(m)
        micro_features.append(feat)
        micro_major_labels.append(mj)
        micro_member_indices.append(idx_m)
        micro_border_indices.append(boundary_global_idx)

    micro_features = np.stack(micro_features, axis=0)  # [K_eff, d + 4]
    micro_major_labels = np.array(micro_major_labels, dtype=int)

    print(f"[INFO] Effective non-empty micro-clusters: {len(micro_features)}")
    return {
        "kmeans": km,
        "train_micro_orig": train_micro_orig,
        "micro_features": micro_features,
        "micro_major_labels": micro_major_labels,
        "orig_to_new": orig_to_new,
        "new_to_orig": new_to_orig,
        "micro_member_indices": micro_member_indices,
        "micro_border_indices": micro_border_indices,
    }


def build_edges_with_pair_features(
    micro_features: np.ndarray,
    micro_border_indices,
    X_train: np.ndarray,
    dist_quantile: float = 0.3,
):
    """
    Build edges between micro-cluster centroids based on distance threshold,
    and compute pairwise features:
      - center_dist
      - border_min_dist (between boundary points)
      - log_density_diff
      - log_density_i
      - log_density_j

    micro_features: [K_eff, d+4], where last 4 dims are [log_size, radius_mean, radius_std, log_density].
    """
    d = X_train.shape[1]
    centroids = micro_features[:, :d]  # (K_eff, d)
    log_density = micro_features[:, d + 3]  # (K_eff,) last scalar feature

    D = pairwise_distances(centroids, metric="euclidean")
    K_eff = centroids.shape[0]
    iu = np.triu_indices(K_eff, k=1)
    dvals = D[iu]
    thresh = np.quantile(dvals, dist_quantile)

    edge_list = []
    pair_extra = []

    for i, j in zip(iu[0], iu[1]):
        center_dist = D[i, j]
        if center_dist > thresh:
            continue

        # Border-to-border min distance
        idx_i = micro_border_indices[i]
        idx_j = micro_border_indices[j]
        pts_i = X_train[idx_i]
        pts_j = X_train[idx_j]
        if len(pts_i) == 0 or len(pts_j) == 0:
            border_min_dist = center_dist
        else:
            Dij = pairwise_distances(pts_i, pts_j)
            border_min_dist = float(Dij.min())

        ld_i = float(log_density[i])
        ld_j = float(log_density[j])
        ld_diff = float(abs(ld_i - ld_j))

        edge_list.append((i, j))
        # distances + density features
        pair_extra.append((center_dist, border_min_dist, ld_diff, ld_i, ld_j))

    edge_list = np.array(edge_list, dtype=int)
    pair_extra = np.array(pair_extra, dtype=np.float32)
    print(f"[INFO] Built {len(edge_list)} edges with dist <= {thresh:.4f} (quantile={dist_quantile})")
    return edge_list, thresh, pair_extra


def plot_micro_graph(micro_features, micro_major_labels, edge_list, dist_thresh, X_dim: int):
    centroids = micro_features[:, :X_dim]
    if X_dim >= 2:
        x = centroids[:, 0]
        y = centroids[:, 1]
    elif X_dim == 1:
        x = centroids[:, 0]
        y = np.zeros_like(x)
    else:
        raise ValueError("X_dim must be >= 1")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(
        x,
        y,
        c=micro_major_labels,
        s=60,
        cmap="tab10",
        edgecolors="black",
        alpha=0.9,
    )
    for (i, j) in edge_list:
        x1, y1 = x[i], y[i]
        x2, y2 = x[j], y[j]
        ax.plot([x1, x2], [y1, y2], linestyle="-", linewidth=0.5, alpha=0.5, color="gray")

    ax.set_title(f"Micro-cluster graph (dist <= {dist_thresh:.3f})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    return fig, ax


# =================== TRANSFORMER-BASED PAIR SCORER =================

class PairTransformerNet(nn.Module):
    def __init__(self, input_dim, d_model=32, n_heads=4, hidden_dim=64, n_layers=2, pair_extra_dim=0):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        in_dim = 4 * d_model + pair_extra_dim
        self.pair_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_features, pair_idx, pair_extra=None):
        """
        node_features: [K_eff, input_dim]
        pair_idx:      [M, 2] int64
        pair_extra:    [M, E] float32 or None
        """
        h0 = self.proj(node_features)  # [K, d_model]
        h_enc = self.encoder(h0.unsqueeze(0))  # [1, K, d_model]
        h = h_enc.squeeze(0)  # [K, d_model]

        hi = h[pair_idx[:, 0]]
        hj = h[pair_idx[:, 1]]

        # Symmetric pair embedding
        z = torch.cat([
            hi,
            hj,
            torch.abs(hi - hj),
            hi * hj,
        ], dim=1)

        if pair_extra is not None:
            z = torch.cat([z, pair_extra], dim=1)

        logits = self.pair_mlp(z).squeeze(1)  # [M]
        return logits


def train_pair_transformer(
    micro_features: np.ndarray,
    micro_major_labels: np.ndarray,
    edge_list: np.ndarray,
    pair_extra: np.ndarray,
    device,
    args,
    dataset: str,
    out_dir: Path,
    figures_dir: Path,
) -> PairTransformerNet:
    """
    Train the PairTransformerNet and save the best model checkpoint under:
      out_dir/checkpoints/gothic_pair_transformer_best_<dataset>.pt

    Also saves training curves and pair-score histograms under:
      out_dir/figures/.
    """
    X_nodes = torch.tensor(micro_features, dtype=torch.float32, device=device)
    if len(edge_list) == 0:
        raise RuntimeError("Edge list is empty; cannot train pair model.")

    pair_idx = torch.tensor(edge_list, dtype=torch.long, device=device)
    pair_extra_tensor = torch.tensor(pair_extra, dtype=torch.float32, device=device)

    # Labels: 1 if same majority cluster, else 0
    labels_np = (micro_major_labels[edge_list[:, 0]] == micro_major_labels[edge_list[:, 1]]).astype(np.float32)
    y_pairs = torch.tensor(labels_np, dtype=torch.float32, device=device)

    model = PairTransformerNet(
        input_dim=micro_features.shape[1],
        d_model=args.d_model,
        n_heads=args.n_heads,
        hidden_dim=args.attn_hidden,
        n_layers=args.n_transformer_layers,
        pair_extra_dim=pair_extra.shape[1],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_acc = -1.0
    best_loss = float("inf")
    best_state = None

    loss_history = []
    acc_history = []

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

        # Best-model tracking
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

    # Restore best weights and save them
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"gothic_pair_transformer_best_{dataset}.pt"

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, ckpt_path)
        print(f"[INFO] Saved best PairTransformerNet weights to: {ckpt_path}")
    else:
        torch.save(model.state_dict(), ckpt_path)
        print("[WARN] No best_state recorded; saved last-epoch model instead at:", ckpt_path)

    # === Training curves plot (loss & accuracy) ===
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
        train_curve_path = figures_dir / f"{dataset}_pair_training_curves.png"
        fig.savefig(train_curve_path, dpi=150, transparent=True)
        print(f"[INFO] Saved training curves figure to: {train_curve_path}")
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] Failed to save training curves plot: {e}")

    # === Pairwise probability histogram (same vs different majority label) ===
    try:
        model.eval()
        with torch.no_grad():
            logits_best = model(X_nodes, pair_idx, pair_extra_tensor)
            probs_best = torch.sigmoid(logits_best).cpu().numpy()
        y_np = y_pairs.cpu().numpy()

        # Basic debug info
        n_total = len(probs_best)
        n_pos = int((y_np == 1).sum())
        n_neg = int((y_np == 0).sum())
        print(f"[INFO] Pair scores: total={n_total}, same-label={n_pos}, diff-label={n_neg}")

        fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))

        # Panel 1: all probabilities
        axes2[0].hist(probs_best, bins=30)
        axes2[0].set_title("All pairwise merge probabilities")
        axes2[0].set_xlabel("P(merge)")
        axes2[0].set_ylabel("Count")
        axes2[0].set_xlim(0.0, 1.0)

        # Panel 2: split by same/different (only plot if non-empty)
        pos = probs_best[y_np == 1]
        neg = probs_best[y_np == 0]

        if len(pos) > 0:
            axes2[1].hist(pos, bins=30, alpha=0.6, label=f"Same label (n={len(pos)})")
        if len(neg) > 0:
            axes2[1].hist(neg, bins=30, alpha=0.6, label=f"Different label (n={len(neg)})")

        if len(pos) == 0 and len(neg) == 0:
            axes2[1].text(
                0.5, 0.5,
                "No edges / no labels",
                ha="center",
                va="center",
                transform=axes2[1].transAxes,
            )

        axes2[1].set_title("P(merge) by majority-label relation")
        axes2[1].set_xlabel("P(merge)")
        axes2[1].set_ylabel("Count")
        axes2[1].set_xlim(0.0, 1.0)
        axes2[1].legend()

        fig2.tight_layout()
        pair_hist_path = figures_dir / f"{dataset}_pair_score_hist.png"
        fig2.savefig(pair_hist_path, dpi=150, transparent=True)
        print(f"[INFO] Saved pair-score histogram to: {pair_hist_path}")
        plt.close(fig2)
    except Exception as e:
        print(f"[WARN] Failed to save pair-score histogram: {e}")

    return model


def merge_micro_clusters_learned(
    micro_features: np.ndarray,
    edge_list: np.ndarray,
    pair_extra: np.ndarray,
    model: PairTransformerNet,
    device,
    k_target: int,
) -> np.ndarray:
    """
    Use learned pair probabilities on edges to greedily merge until k_target.
    If still more than k_target sets, fallback to distance-based merges.
    """
    K_eff = micro_features.shape[0]
    uf = UnionFind(K_eff)

    X_nodes = torch.tensor(micro_features, dtype=torch.float32, device=device)
    pair_idx = torch.tensor(edge_list, dtype=torch.long, device=device)
    pair_extra_tensor = torch.tensor(pair_extra, dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        logits = model(X_nodes, pair_idx, pair_extra_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()

    order = np.argsort(-probs)  # descending by probability
    for idx in order:
        i, j = edge_list[idx]
        if uf.set_count <= k_target:
            break
        uf.union(i, j)

    print(f"[INFO] After learned merges, components = {uf.set_count}")

    if uf.set_count > k_target:
        print("[INFO] Falling back to distance-based merges to reach k_target.")
        d = X_nodes.shape[1]
        centroids = micro_features[:, :d]
        D = pairwise_distances(centroids)
        iu = np.triu_indices(K_eff, k=1)
        dist_pairs = list(zip(iu[0], iu[1], D[iu]))
        dist_pairs.sort(key=lambda t: t[2])  # ascending distance
        for i, j, d_ij in dist_pairs:
            if uf.set_count <= k_target:
                break
            uf.union(i, j)

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
    # Nest run-specific outputs under out_dir/run_id/
    root_out_dir = Path(args.out_dir)
    out_dir = root_out_dir / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Outputs will be saved under: {out_dir}")
    print(f"[INFO] Figures will be saved under: {figures_dir}")

    # Determine k_target if not supplied
    if args.k_target is None or args.k_target <= 0:
        k_target = DEFAULT_K_TARGETS.get(dataset, 6)
        print(f"[INFO] Using dataset-specific k_target={k_target} for dataset={dataset}")
    else:
        k_target = args.k_target
        print(f"[INFO] Using user-specified k_target={k_target}")

    # Step 1: Load dataset (loader will ensure files exist)
    X, y = load_dataset(dataset, base_dir=str(data_root))
    print(f"[INFO] Loaded dataset '{dataset}': X.shape={X.shape}, unique labels={np.unique(y)}")

    # Step 2: Train/test split
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split_with_strategy(
        X, y,
        train_frac=args.train_fraction,
        strategy=args.split_strategy,
        seed=args.seed,
    )
    print(f"[INFO] Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Step 3: Over-cluster on train
    mc = build_micro_clusters(
        X_train,
        y_train,
        n_micro=args.n_micro,
        seed=args.seed,
        kmeans_n_init=args.kmeans_n_init,
        kmeans_max_iter=args.kmeans_max_iter,
        b_boundary=args.b_boundary,
    )
    micro_features = mc["micro_features"]         # (K_eff, d+4)
    micro_major_labels = mc["micro_major_labels"] # (K_eff,)
    orig_to_new = mc["orig_to_new"]
    new_to_orig = mc["new_to_orig"]
    km = mc["kmeans"]
    train_micro_orig = mc["train_micro_orig"]
    micro_border_indices = mc["micro_border_indices"]

    train_micro_new = np.array([orig_to_new[m] for m in train_micro_orig], dtype=int)

    # --- Extra plots: micro-cluster scalar feature histograms ---
    try:
        d = X_train.shape[1]
        scalar_feats = micro_features[:, d:]  # [K_eff, 4]
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

    # Step 4: Build graph + pair features + plot
    edge_list, dist_thresh, pair_extra = build_edges_with_pair_features(
        micro_features,
        micro_border_indices,
        X_train,
        dist_quantile=args.dist_quantile,
    )

    # --- Extra plots: graph degree histogram & edge feature histograms + distance heatmap ---
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

    try:
        # Edge feature histograms: distances + density-related features
        num_extra = pair_extra.shape[1]
        if num_extra >= 3:
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

        if num_extra >= 5:
            fig_edge2, axes_edge2 = plt.subplots(1, 2, figsize=(10, 4))
            labels_edge2 = ["log_density_i", "log_density_j"]
            for i in range(2):
                axes_edge2[i].hist(pair_extra[:, 3 + i], bins=30)
                axes_edge2[i].set_title(labels_edge2[i])
                axes_edge2[i].set_xlabel(labels_edge2[i])
                axes_edge2[i].set_ylabel("Count")
            fig_edge2.tight_layout()
            edge_hist_path2 = figures_dir / f"{dataset}_edge_density_features_hist.png"
            fig_edge2.savefig(edge_hist_path2, dpi=150, transparent=True)
            print(f"[INFO] Saved edge density-feature histograms to: {edge_hist_path2}")
            if args.show_plots:
                fig_edge2.show()
            else:
                plt.close(fig_edge2)
    except Exception as e:
        print(f"[WARN] Failed to save edge-feature histograms: {e}")

    try:
        # Heatmap of pairwise micro-centroid distances
        d = X_train.shape[1]
        centroids_eff = micro_features[:, :d]
        D_micro = pairwise_distances(centroids_eff)
        fig_heat, ax_heat = plt.subplots(figsize=(6, 5))
        im = ax_heat.imshow(D_micro, interpolation="nearest", aspect="auto")
        fig_heat.colorbar(im, ax=ax_heat)
        ax_heat.set_title("Pairwise micro-centroid distance matrix")
        ax_heat.set_xlabel("Micro-cluster index")
        ax_heat.set_ylabel("Micro-cluster index")
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

    # Main micro-graph visualization
    fig_graph, _ = plot_micro_graph(
        micro_features,
        micro_major_labels,
        edge_list,
        dist_thresh,
        X_dim=X_train.shape[1],
    )
    graph_path = figures_dir / f"{dataset}_micro_graph.png"
    fig_graph.savefig(graph_path, dpi=150, transparent=True)
    print(f"[INFO] Saved micro-graph figure to: {graph_path}")
    if args.show_plots:
        fig_graph.show()
    else:
        plt.close(fig_graph)

    # Step 5: Train Transformer-based pair scorer (supervised on micro-major labels)
    model = train_pair_transformer(
        micro_features,
        micro_major_labels,
        edge_list,
        pair_extra,
        device,
        args,
        dataset=dataset,
        out_dir=out_dir,
        figures_dir=figures_dir,
    )

    # Step 6: Merge micro-clusters using learned scores
    micro_to_macro = merge_micro_clusters_learned(
        micro_features,
        edge_list,
        pair_extra,
        model,
        device,
        k_target=k_target,
    )

    # Step 7: Label train & test, compute metrics

    # Train predictions
    y_pred_train_macro = np.zeros_like(y_train)
    for i in range(len(X_train)):
        m_new = train_micro_new[i]
        y_pred_train_macro[i] = micro_to_macro[m_new]

    # Test predictions: nearest effective micro-centroid
    centroids_all = km.cluster_centers_                 # (n_micro, d)
    centroids_eff = centroids_all[new_to_orig]          # (K_eff, d)
    dtest = pairwise_distances(X_test, centroids_eff)
    nearest_micro_eff = dtest.argmin(axis=1)
    y_pred_test_macro = micro_to_macro[nearest_micro_eff]

    # Core clustering metrics
    nmi_train = normalized_mutual_info_score(y_train, y_pred_train_macro)
    nmi_test = normalized_mutual_info_score(y_test, y_pred_test_macro)

    ami_train = adjusted_mutual_info_score(y_train, y_pred_train_macro)
    ami_test = adjusted_mutual_info_score(y_test, y_pred_test_macro)

    ari_train = adjusted_rand_score(y_train, y_pred_train_macro)
    ari_test = adjusted_rand_score(y_test, y_pred_test_macro)

    fmi_train = fowlkes_mallows_score(y_train, y_pred_train_macro)
    fmi_test = fowlkes_mallows_score(y_test, y_pred_test_macro)

    # Accuracy with best permutation
    acc_train, map_train = clustering_accuracy_with_map(y_train, y_pred_train_macro)
    acc_test, map_test = clustering_accuracy_with_map(y_test, y_pred_test_macro)

    # Intrinsic metrics (no labels; purely geometry-based on predicted clusters)
    def safe_cluster_metric(fn, X_data, labels_pred, name):
        try:
            if len(np.unique(labels_pred)) < 2:
                return float("nan")
            return float(fn(X_data, labels_pred))
        except Exception as e:
            print(f"[WARN] Failed to compute {name}: {e}")
            return float("nan")

    sil_train = safe_cluster_metric(silhouette_score, X_train, y_pred_train_macro, "silhouette (train)")
    sil_test = safe_cluster_metric(silhouette_score, X_test, y_pred_test_macro, "silhouette (test)")

    db_train = safe_cluster_metric(davies_bouldin_score, X_train, y_pred_train_macro, "Davies-Bouldin (train)")
    db_test = safe_cluster_metric(davies_bouldin_score, X_test, y_pred_test_macro, "Davies-Bouldin (test)")

    ch_train = safe_cluster_metric(calinski_harabasz_score, X_train, y_pred_train_macro, "Calinski-Harabasz (train)")
    ch_test = safe_cluster_metric(calinski_harabasz_score, X_test, y_pred_test_macro, "Calinski-Harabasz (test)")

    print(f"[RESULT] Dataset = {dataset}")
    print(f"[RESULT] Train NMI   = {nmi_train:.4f}, ACC = {acc_train:.4f}")
    print(f"[RESULT] Train AMI   = {ami_train:.4f}, ARI = {ari_train:.4f}, FMI = {fmi_train:.4f}")
    print(f"[RESULT] Train Sil   = {sil_train:.4f}, DB  = {db_train:.4f}, CH  = {ch_train:.4f}")
    print(f"[RESULT] Test  NMI   = {nmi_test:.4f}, ACC = {acc_test:.4f}")
    print(f"[RESULT] Test  AMI   = {ami_test:.4f}, ARI = {ari_test:.4f}, FMI = {fmi_test:.4f}")
    print(f"[RESULT] Test  Sil   = {sil_test:.4f}, DB  = {db_test:.4f}, CH  = {ch_test:.4f}")

    # For test visualization: map predicted labels to best-aligned true labels
    y_pred_test_mapped = apply_label_mapping(y_pred_test_macro, map_test)
    correct_test = (y_pred_test_mapped == y_test)

    # ==================== PLOTS: TRAIN + TEST ======================

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Decide 2D coordinates for plotting: use first two features
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

    # (b) Train predicted clusters (title just NMI & ACC)
    ax = axes[1]
    ax.scatter(
        train_x,
        train_y,
        c=y_pred_train_macro,
        cmap="tab10",
        s=20,
        alpha=0.9,
    )
    ax.set_title(f"Train predicted (NMI={nmi_train:.3f}, ACC={acc_train:.3f})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # (c) Test: blue baseline, green correct, red mis-clustered
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
    ax.set_title(f"Test (NMI={nmi_test:.3f}, ACC={acc_test:.3f})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()

    results_fig_path = figures_dir / f"{dataset}_train_test.png"
    fig.savefig(results_fig_path, dpi=150, transparent=True)
    print(f"[INFO] Saved train/test results figure to: {results_fig_path}")
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

        full_gt_path = figures_dir / f"{dataset}_full_ground_truth.png"
        fig_full.savefig(full_gt_path, dpi=150, transparent=True)
        print(f"[INFO] Saved full dataset ground-truth figure to: {full_gt_path}")
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
            ax_m.set_title(f"{name} (train vs test)")
            ax_m.set_ylabel(name)
            fig_m.tight_layout()
            metric_path = figures_dir / f"{dataset}_metric_{name.lower()}.png"
            fig_m.savefig(metric_path, dpi=150, transparent=True)
            print(f"[INFO] Saved metric plot '{name}' to: {metric_path}")
            if args.show_plots:
                fig_m.show()
            else:
                plt.close(fig_m)
    except Exception as e:
        print(f"[WARN] Failed to save metric bar plots: {e}")

    # ================ SAVE TEXT RESULTS / METRICS ==================

    results_txt_path = out_dir / f"{dataset}_results.txt"
    with open(results_txt_path, "w") as f:
        f.write(f"dataset          = {dataset}\n")
        f.write(f"run_id           = {args.run_id}\n")
        f.write(f"seed             = {args.seed}\n")
        f.write(f"split_strategy   = {args.split_strategy}\n")
        f.write(f"train_fraction   = {args.train_fraction}\n")
        f.write(f"n_micro          = {args.n_micro}\n")
        f.write(f"k_target         = {k_target}\n")
        f.write(f"dist_quantile    = {args.dist_quantile}\n")
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
    }


if __name__ == "__main__":
    gothic_main()
