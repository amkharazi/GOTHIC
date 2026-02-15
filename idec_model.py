"""
idec_model.py

IDEC (Improved Deep Embedded Clustering) baseline.
- PyTorch MLP AutoEncoder
- KMeans init of cluster centers in latent space
- Fine-tune with (reconstruction + gamma * KL(P || Q))

This version matches Spectral's output style:
  - writes results txt to out_dir/run_id/
  - saves figures to out_dir/run_id/figures/
  - appends a row to results_summary.csv in Path.cwd()
"""

MODEL_NAME = "idec"

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
    p = argparse.ArgumentParser(description="IDEC deep clustering baseline.")

    p.add_argument("--dataset", type=str, default="compound")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_fraction", type=float, default=0.8)
    p.add_argument("--split_strategy", type=str, default="balanced", choices=["random", "balanced"])
    p.add_argument("--data_root", type=str, default="datasets")
    p.add_argument("--out_dir", type=str, default="outputs")
    p.add_argument("--run_id", type=str, default="idec_default")

    p.add_argument("--k_target", type=int, default=-1)

    # AE / IDEC
    p.add_argument("--latent_dim", type=int, default=16)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--pretrain_epochs", type=int, default=100)
    p.add_argument("--finetune_epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)

    p.add_argument("--alpha", type=float, default=1.0, help="Student-t alpha.")
    p.add_argument("--gamma", type=float, default=1.0, help="Weight on KL term (clustering).")
    p.add_argument("--update_interval", type=int, default=10, help="How often to recompute target distribution P (epochs).")

    p.add_argument("--plot_pca", action="store_true")
    p.add_argument("--show_plots", action="store_true")

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


# =========================== UTILITIES ============================

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

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx], train_idx, test_idx


def to_2d_for_plot(X: np.ndarray, use_pca: bool) -> np.ndarray:
    if X.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if X.shape[1] >= 2 and not use_pca:
        return X[:, :2]
    if X.shape[1] == 1:
        return np.column_stack([X[:, 0], np.zeros_like(X[:, 0])])
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(X)


def safe_cluster_metric(fn, X_data, labels_pred, name: str) -> float:
    try:
        if len(np.unique(labels_pred)) < 2:
            return float("nan")
        return float(fn(X_data, labels_pred))
    except Exception as e:
        print(f"[WARN] Failed to compute {name}: {e}")
        return float("nan")


# =========================== IDEC MODEL ============================

class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class MLPDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z):
        return self.net(z)


class AutoEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int, dropout: float):
        super().__init__()
        self.encoder = MLPEncoder(in_dim, hidden_dim, latent_dim, dropout)
        self.decoder = MLPDecoder(latent_dim, hidden_dim, in_dim, dropout)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat


def student_t_q(z: torch.Tensor, centers: torch.Tensor, alpha: float) -> torch.Tensor:
    dist2 = torch.cdist(z, centers) ** 2
    num = (1.0 + dist2 / alpha) ** (-(alpha + 1.0) / 2.0)
    q = num / torch.sum(num, dim=1, keepdim=True)
    return q


def target_p_from_q(q: torch.Tensor) -> torch.Tensor:
    f = torch.sum(q, dim=0)  # (K,)
    p = (q ** 2) / (f.unsqueeze(0) + 1e-12)
    p = p / torch.sum(p, dim=1, keepdim=True)
    return p


# ============================= MAIN ================================

def idec_main(args=None) -> Dict[str, Any]:
    if args is None:
        args = parse_args()

    dataset = args.dataset.lower()
    set_seed(args.seed)

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

    # Determine k_target
    if args.k_target is None or args.k_target <= 0:
        k_target = DEFAULT_K_TARGETS.get(dataset, 6)
        print(f"[INFO] Using dataset-specific k_target={k_target} for dataset={dataset}")
    else:
        k_target = args.k_target
        print(f"[INFO] Using user-specified k_target={k_target}")

    # Load
    X, y = load_dataset(dataset, base_dir=str(data_root))
    X = X.astype(np.float32)
    y = y.astype(int)
    print(f"[INFO] Loaded dataset '{dataset}': X.shape={X.shape}, unique labels={np.unique(y)}")

    # Split
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split_with_strategy(
        X, y,
        train_frac=args.train_fraction,
        strategy=args.split_strategy,
        seed=args.seed,
    )
    print(f"[INFO] Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Tensors
    Xtr = torch.tensor(X_train, dtype=torch.float32, device=device)
    Xte = torch.tensor(X_test, dtype=torch.float32, device=device) if len(X_test) else None

    # Model
    ae = AutoEncoder(
        in_dim=X_train.shape[1],
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        dropout=args.dropout
    ).to(device)

    opt = torch.optim.Adam(ae.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ------------------- PRETRAIN AE -------------------
    ae.train()
    bs = max(16, int(args.batch_size))
    n = Xtr.shape[0]

    print("[INFO] Pretraining autoencoder ...")
    for ep in range(1, args.pretrain_epochs + 1):
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = Xtr[idx]
            _, xhat = ae(xb)
            loss = F.mse_loss(xhat, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if ep == 1 or ep % 20 == 0 or ep == args.pretrain_epochs:
            print(f"[AE] pretrain epoch {ep:4d}/{args.pretrain_epochs} | mse={total/n:.6f}")

    # Initial embeddings
    ae.eval()
    with torch.no_grad():
        Ztr, _ = ae(Xtr)
        Ztr_np = Ztr.detach().cpu().numpy()

    # ------------------- INIT CENTERS (KMeans) -------------------
    km_init = KMeans(n_clusters=k_target, n_init=20, random_state=args.seed)
    km_init.fit(Ztr_np)
    centers = torch.tensor(km_init.cluster_centers_, dtype=torch.float32, device=device, requires_grad=True)

    opt2 = torch.optim.Adam(list(ae.parameters()) + [centers], lr=args.lr, weight_decay=args.weight_decay)

    with torch.no_grad():
        q = student_t_q(Ztr, centers, alpha=args.alpha)
        p = target_p_from_q(q)

    # ------------------- FINETUNE IDEC -------------------
    print("[INFO] Fine-tuning IDEC (recon + gamma * KL(P||Q)) ...")
    ae.train()
    for ep in range(1, args.finetune_epochs + 1):
        if ep == 1 or (args.update_interval > 0 and ep % args.update_interval == 0):
            ae.eval()
            with torch.no_grad():
                Ztr, _ = ae(Xtr)
                q = student_t_q(Ztr, centers, alpha=args.alpha)
                p = target_p_from_q(q)
            ae.train()

        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0

        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = Xtr[idx]

            z, xhat = ae(xb)
            q_b = student_t_q(z, centers, alpha=args.alpha)
            p_b = p[idx]

            recon = F.mse_loss(xhat, xb)
            kl = F.kl_div((q_b + 1e-12).log(), p_b, reduction="batchmean")
            loss = recon + args.gamma * kl

            opt2.zero_grad()
            loss.backward()
            opt2.step()

            total_loss += loss.item() * len(idx)
            total_recon += recon.item() * len(idx)
            total_kl += kl.item() * len(idx)

        if ep == 1 or ep % 20 == 0 or ep == args.finetune_epochs:
            print(
                f"[IDEC] epoch {ep:4d}/{args.finetune_epochs} | "
                f"loss={total_loss/n:.6f} recon={total_recon/n:.6f} kl={total_kl/n:.6f}"
            )

    # ------------------- PREDICT -------------------
    ae.eval()
    with torch.no_grad():
        Ztr, _ = ae(Xtr)
        qtr = student_t_q(Ztr, centers, alpha=args.alpha)
        y_pred_train = torch.argmax(qtr, dim=1).cpu().numpy().astype(int)
        Ztr_np = Ztr.cpu().numpy()

        if Xte is not None and len(X_test) > 0:
            Zte, _ = ae(Xte)
            qte = student_t_q(Zte, centers, alpha=args.alpha)
            y_pred_test = torch.argmax(qte, dim=1).cpu().numpy().astype(int)
            Zte_np = Zte.cpu().numpy()
        else:
            y_pred_test = np.zeros(len(X_test), dtype=int)
            Zte_np = np.zeros((len(X_test), args.latent_dim), dtype=np.float32)

    # ------------------- METRICS -------------------
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
    else:
        nmi_test = ami_test = ari_test = fmi_test = acc_test = float("nan")
        map_test = {}

    # Note: For IDEC we compute silhouette/DB/CH in embedding space Z
    sil_train = safe_cluster_metric(silhouette_score, Ztr_np, y_pred_train, "silhouette (train,Z)")
    sil_test = safe_cluster_metric(silhouette_score, Zte_np, y_pred_test, "silhouette (test,Z)")
    db_train_val = safe_cluster_metric(davies_bouldin_score, Ztr_np, y_pred_train, "Davies-Bouldin (train,Z)")
    db_test_val = safe_cluster_metric(davies_bouldin_score, Zte_np, y_pred_test, "Davies-Bouldin (test,Z)")
    ch_train = safe_cluster_metric(calinski_harabasz_score, Ztr_np, y_pred_train, "Calinski-Harabasz (train,Z)")
    ch_test = safe_cluster_metric(calinski_harabasz_score, Zte_np, y_pred_test, "Calinski-Harabasz (test,Z)")

    print(f"[RESULT] Dataset = {dataset} (IDEC)")
    print(f"[RESULT] Train NMI={nmi_train:.4f}, ACC={acc_train:.4f}")
    print(f"[RESULT] Train AMI={ami_train:.4f}, ARI={ari_train:.4f}, FMI={fmi_train:.4f}")
    print(f"[RESULT] Train Sil(Z)={sil_train:.4f}, DB(Z)={db_train_val:.4f}, CH(Z)={ch_train:.4f}")
    print(f"[RESULT] Test  NMI={nmi_test:.4f}, ACC={acc_test:.4f}")
    print(f"[RESULT] Test  AMI={ami_test:.4f}, ARI={ari_test:.4f}, FMI={fmi_test:.4f}")
    print(f"[RESULT] Test  Sil(Z)={sil_test:.4f}, DB(Z)={db_test_val:.4f}, CH(Z)={ch_test:.4f}")

    # ------------------- PLOTS (input space, like spectral) -------------------
    Xtr2 = to_2d_for_plot(X_train, use_pca=args.plot_pca or (X_train.shape[1] > 2))
    Xte2 = to_2d_for_plot(X_test, use_pca=args.plot_pca or (X_train.shape[1] > 2))

    if len(y_test) > 0:
        y_pred_test_mapped = apply_label_mapping(y_pred_test, map_test)
        correct_test = (y_pred_test_mapped == y_test)
    else:
        correct_test = np.array([], dtype=bool)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].scatter(Xtr2[:, 0], Xtr2[:, 1], c=y_train, cmap="tab10", s=20, alpha=0.9)
    axes[0].set_title("Train ground truth")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")

    axes[1].scatter(Xtr2[:, 0], Xtr2[:, 1], c=y_pred_train, cmap="tab10", s=20, alpha=0.9)
    axes[1].set_title(f"Train IDEC (NMI={nmi_train:.3f}, ACC={acc_train:.3f})")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("y")

    axes[2].set_title(f"Test IDEC (NMI={nmi_test:.3f}, ACC={acc_test:.3f})")
    axes[2].set_xlabel("x"); axes[2].set_ylabel("y")
    if len(Xte2) > 0:
        axes[2].scatter(Xte2[:, 0], Xte2[:, 1], c="blue", s=20, alpha=0.3, label="Test points")
        if len(correct_test) == len(Xte2):
            axes[2].scatter(Xte2[correct_test, 0], Xte2[correct_test, 1], c="green", s=20, alpha=0.9, label="Correct")
            axes[2].scatter(Xte2[~correct_test, 0], Xte2[~correct_test, 1], c="red", s=30, alpha=0.9, marker="x", label="Mis-clustered")
        axes[2].legend(loc="best", fontsize=8)

    fig.tight_layout()
    train_test_path = figures_dir / f"{dataset}_idec_train_test.png"
    fig.savefig(train_test_path, dpi=150, transparent=True)
    print(f"[INFO] Saved IDEC train/test figure to: {train_test_path}")
    if args.show_plots:
        plt.show()
    else:
        plt.close(fig)

    # ------------------- SAVE RESULTS (spectral-like formatting) -------------------
    results_txt_path = out_dir / f"{dataset}_idec_results.txt"
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
        f.write(f"latent_dim       = {args.latent_dim}\n")
        f.write(f"hidden_dim       = {args.hidden_dim}\n")
        f.write(f"dropout          = {args.dropout}\n")
        f.write(f"batch_size       = {args.batch_size}\n")
        f.write(f"pretrain_epochs  = {args.pretrain_epochs}\n")
        f.write(f"finetune_epochs  = {args.finetune_epochs}\n")
        f.write(f"lr               = {args.lr}\n")
        f.write(f"weight_decay     = {args.weight_decay}\n")
        f.write(f"alpha            = {args.alpha}\n")
        f.write(f"gamma            = {args.gamma}\n")
        f.write(f"update_interval  = {args.update_interval}\n")
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

    print(f"[INFO] Saved IDEC scalar results to: {results_txt_path}")

    # ------------------- APPEND RUN SUMMARY (same as spectral) -------------------
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

        # idec-specific
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "pretrain_epochs": args.pretrain_epochs,
        "finetune_epochs": args.finetune_epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "alpha": args.alpha,
        "gamma": args.gamma,
        "update_interval": args.update_interval,
        "metric_space": "Z",

        # metrics (use same key names as spectral)
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
    idec_main()
