"""
tune_gothic.py

- Runs many configs for GOTHIC
- Parses <out_dir>/<run_id>/<dataset>_results.txt
- Writes a tuning CSV
- IMPORTANT: if a trial fails, prints the tail of stderr/stdout and saves it into CSV
- OPTIONAL: fallback to calling gothic_model.py directly if runner.py fails
"""

from __future__ import annotations

import argparse
import csv
import random
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import re

RE_FLOAT = r"[-+]?\d*\.\d+|\d+"
TXT_TEST_NMI = re.compile(r"Test\s+NMI\s*=\s*(" + RE_FLOAT + r")", re.IGNORECASE)
TXT_TEST_ACC = re.compile(r"Test\s+ACC\s*=\s*(" + RE_FLOAT + r")", re.IGNORECASE)
TXT_TRAIN_NMI = re.compile(r"Train\s+NMI\s*=\s*(" + RE_FLOAT + r")", re.IGNORECASE)
TXT_TRAIN_ACC = re.compile(r"Train\s+ACC\s*=\s*(" + RE_FLOAT + r")", re.IGNORECASE)


def parse_metric_from_results_txt(txt_path: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not txt_path.exists():
        return out
    s = txt_path.read_text(encoding="utf-8", errors="ignore")

    m = TXT_TRAIN_NMI.search(s)
    if m: out["train_nmi"] = float(m.group(1))
    m = TXT_TRAIN_ACC.search(s)
    if m: out["train_acc"] = float(m.group(1))
    m = TXT_TEST_NMI.search(s)
    if m: out["test_nmi"] = float(m.group(1))
    m = TXT_TEST_ACC.search(s)
    if m: out["test_acc"] = float(m.group(1))
    return out


@dataclass
class GothicTrial:
    seed: int = 42
    train_fraction: float = 0.8
    split_strategy: str = "balanced"

    n_micro: int = 120
    dist_quantile: float = 0.4
    b_boundary: int = 5

    d_model: int = 32
    n_heads: int = 4
    attn_hidden: int = 64
    n_transformer_layers: int = 4

    train_epochs: int = 2000
    lr: float = 1e-3
    weight_decay: float = 1e-4

    reduce_dim: str = "none"   # none|pca
    pca_dim: int = 128

    def to_cmd_args(self) -> List[str]:
        args = [
            "--seed", str(self.seed),
            "--train_fraction", str(self.train_fraction),
            "--split_strategy", self.split_strategy,

            "--n_micro", str(self.n_micro),
            "--dist_quantile", str(self.dist_quantile),
            "--b_boundary", str(self.b_boundary),

            "--d_model", str(self.d_model),
            "--n_heads", str(self.n_heads),
            "--attn_hidden", str(self.attn_hidden),
            "--n_transformer_layers", str(self.n_transformer_layers),

            "--train_epochs", str(self.train_epochs),
            "--lr", str(self.lr),
            "--weight_decay", str(self.weight_decay),
        ]

        if self.reduce_dim and self.reduce_dim.lower() != "none":
            args += ["--reduce_dim", self.reduce_dim]
            if self.reduce_dim.lower() == "pca":
                args += ["--pca_dim", str(int(self.pca_dim))]

        return args


def _micro_candidates(k_target: int) -> List[int]:
    base = sorted(set([
        k_target,
        int(round(1.25 * k_target)),
        int(round(1.5 * k_target)),
        int(round(2.0 * k_target)),
        int(round(3.0 * k_target)),
        int(round(4.0 * k_target)),
    ]))
    extra = [60, 80, 100, 120, 160, 200, 240, 300]
    out = sorted(set([m for m in (base + extra) if m >= k_target and m <= 400]))
    return out or [k_target]


def sample_trial(
    rng: random.Random,
    *,
    base_seed: int,
    k_target: int,
    train_fraction: float,
    split_strategy: str,
    reduce_dim: str,
    pca_dim: int,
) -> GothicTrial:
    n_micro = rng.choice(_micro_candidates(k_target))
    dist_q = rng.choice([0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    b_boundary = rng.choice([3, 5, 8, 10, 12, 15, 20])

    d_model = rng.choice([16, 32, 64, 96, 128, 256])
    possible_heads = [h for h in [1, 2, 4, 8] if d_model % h == 0]
    n_heads = rng.choice(possible_heads) if possible_heads else 1
    attn_hidden = rng.choice([32, 64, 128, 256])
    n_layers = rng.choice([1, 2, 3, 4, 6])

    epochs = rng.choice([800, 1200, 2000, 3000, 4000])
    lr = rng.choice([3e-4, 5e-4, 1e-3, 2e-3, 3e-3])
    wd = rng.choice([0.0, 1e-6, 1e-5, 1e-4, 5e-4, 1e-3])

    return GothicTrial(
        seed=base_seed,
        train_fraction=train_fraction,
        split_strategy=split_strategy,
        n_micro=max(n_micro, k_target),
        dist_quantile=dist_q,
        b_boundary=b_boundary,
        d_model=d_model,
        n_heads=n_heads,
        attn_hidden=attn_hidden,
        n_transformer_layers=n_layers,
        train_epochs=epochs,
        lr=lr,
        weight_decay=wd,
        reduce_dim=reduce_dim,
        pca_dim=pca_dim,
    )


def append_csv(csv_path: Path, row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def _tail(s: str, n: int = 4000) -> str:
    s = s or ""
    return s[-n:] if len(s) > n else s


def _run_cmd(cmd: List[str], verbose: bool) -> Tuple[bool, str]:
    if verbose:
        print("\n[CMD]", " ".join(cmd))
        proc = subprocess.run(cmd, check=False)
        return (proc.returncode == 0), f"returncode={proc.returncode}"
    else:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        log = (proc.stdout or "") + "\n" + (proc.stderr or "")
        return (proc.returncode == 0), _tail(log, 4000)


def run_one(
    *,
    dataset: str,
    k_target: int,
    data_root: str,
    out_dir: Path,
    run_prefix: str,
    trial_id: int,
    trial: GothicTrial,
    runner_py: str,
    gothic_py: str,
    python_exe: str,
    kmeans_n_init: int,
    kmeans_max_iter: int,
    verbose: bool,
    fallback_direct: bool,
) -> Tuple[Optional[float], Dict[str, float], str, str]:
    run_id = f"{run_prefix}_T{trial_id:04d}"

    # --- primary: runner.py ---
    cmd_runner = [
        python_exe, runner_py,
        "--model", "gothic",
        "--run_id", run_id,
        "--dataset", dataset,
        "--k_target", str(k_target),
        "--data_root", data_root,
        "--out_dir", str(out_dir),
        "--kmeans_n_init", str(kmeans_n_init),
        "--kmeans_max_iter", str(kmeans_max_iter),
        *trial.to_cmd_args(),
    ]

    ok, log_tail = _run_cmd(cmd_runner, verbose=verbose)
    mode = "runner"

    # --- fallback: call gothic_model.py directly (avoids runner arg limitations) ---
    if (not ok) and fallback_direct:
        mode = "direct_gothic"
        cmd_direct = [
            python_exe, gothic_py,
            "--run_id", run_id,
            "--dataset", dataset,
            "--k_target", str(k_target),
            "--data_root", data_root,
            "--out_dir", str(out_dir),
            "--kmeans_n_init", str(kmeans_n_init),
            "--kmeans_max_iter", str(kmeans_max_iter),
            *trial.to_cmd_args(),
        ]
        ok2, log2 = _run_cmd(cmd_direct, verbose=verbose)
        ok = ok2
        log_tail = log2 if not ok2 else log_tail

    if not ok:
        return None, {}, mode, log_tail

    results_txt = out_dir / run_id / f"{dataset}_results.txt"
    metrics = parse_metric_from_results_txt(results_txt)
    test_nmi = metrics.get("test_nmi", None)

    # if it ran but metrics missing, treat as failure (and expose why)
    if test_nmi is None:
        return None, metrics, mode, f"Run completed but could not parse metrics from: {results_txt}"

    return test_nmi, metrics, mode, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="wine")
    ap.add_argument("--k_target", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_fraction", type=float, default=0.8)
    ap.add_argument("--split_strategy", type=str, default="balanced")
    ap.add_argument("--data_root", type=str, default="datasets")
    ap.add_argument("--out_dir", type=str, default="outputs/gothic/real")

    ap.add_argument("--runner_py", type=str, default="runner.py")
    ap.add_argument("--gothic_py", type=str, default="gothic_model.py")
    ap.add_argument("--python_exe", type=str, default=sys.executable)

    ap.add_argument("--kmeans_n_init", type=int, default=50)
    ap.add_argument("--kmeans_max_iter", type=int, default=500)

    ap.add_argument("--max_trials", type=int, default=80)
    ap.add_argument("--start_trial", type=int, default=1)
    ap.add_argument("--run_prefix", type=str, default="OF")

    ap.add_argument("--reduce_dim", type=str, default="none", choices=["none", "pca"])
    ap.add_argument("--pca_dim", type=int, default=128)

    ap.add_argument("--stop_at", type=float, default=0.9999)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--fallback_direct", action="store_true",
                    help="If runner.py fails, retry by calling gothic_model.py directly.")
    args = ap.parse_args()

    dataset = args.dataset.lower()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tuner_csv = out_dir / f"{dataset}_gothic_tuning.csv"

    rng = random.Random(12345)

    best_nmi = -1.0
    best_row: Optional[Dict[str, Any]] = None

    print(f"[TUNER] dataset={dataset} k_target={args.k_target} seed={args.seed} trials={args.max_trials}")
    print(f"[TUNER] reduce_dim={args.reduce_dim} pca_dim={args.pca_dim}")
    print(f"[TUNER] writing: {tuner_csv}")

    for t in range(args.start_trial, args.start_trial + args.max_trials):
        trial = sample_trial(
            rng,
            base_seed=args.seed,
            k_target=args.k_target,
            train_fraction=args.train_fraction,
            split_strategy=args.split_strategy,
            reduce_dim=args.reduce_dim,
            pca_dim=args.pca_dim,
        )

        test_nmi, metrics, mode, err = run_one(
            dataset=dataset,
            k_target=args.k_target,
            data_root=args.data_root,
            out_dir=out_dir,
            run_prefix=args.run_prefix,
            trial_id=t,
            trial=trial,
            runner_py=args.runner_py,
            gothic_py=args.gothic_py,
            python_exe=args.python_exe,
            kmeans_n_init=args.kmeans_n_init,
            kmeans_max_iter=args.kmeans_max_iter,
            verbose=args.verbose,
            fallback_direct=args.fallback_direct,
        )

        row = {
            "trial": t,
            "dataset": dataset,
            "k_target": args.k_target,
            "exec_mode": mode,
            **asdict(trial),
            "test_nmi": "" if test_nmi is None else test_nmi,
            "train_nmi": metrics.get("train_nmi", ""),
            "test_acc": metrics.get("test_acc", ""),
            "train_acc": metrics.get("train_acc", ""),
            "status": "ok" if test_nmi is not None else "fail",
            "error_tail": err,
        }
        append_csv(tuner_csv, row)

        if test_nmi is None:
            print(f"[TUNER] trial {t:04d} FAILED (mode={mode})")
            if err:
                print("------ error tail ------")
                print(err)
                print("------------------------")
            continue

        if test_nmi > best_nmi:
            best_nmi = test_nmi
            best_row = row

        print(
            f"[TUNER] trial {t:04d} test_nmi={test_nmi:.4f} (best={best_nmi:.4f}) "
            f"n_micro={trial.n_micro} q={trial.dist_quantile} b={trial.b_boundary} "
            f"layers={trial.n_transformer_layers} d_model={trial.d_model} heads={trial.n_heads} "
            f"lr={trial.lr} wd={trial.weight_decay} "
            f"{trial.reduce_dim}{'='+str(trial.pca_dim) if trial.reduce_dim=='pca' else ''}"
        )

        if test_nmi >= args.stop_at:
            print(f"[TUNER] STOP: reached test_nmi >= {args.stop_at}")
            break

    print("\n[TUNER] DONE.")
    if best_row:
        print("[TUNER] BEST CONFIG saved in CSV.")
    else:
        print("[TUNER] No successful runs. Check printed error tails and CSV error_tail column.")


if __name__ == "__main__":
    main()
