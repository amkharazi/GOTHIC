# ablation_study.py
# Run from the repo root (where runner.py exists):
#   python ablation_study.py
#
# It will:
#  - Run seed sweep (confidence) for wine using your baseline args
#  - Run one-factor ablations at seed=42
#  - Save CSVs + transparent plots under: <out_dir>/ABlationReports/<dataset>/<timestamp>/

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Baseline config (YOUR example)
# -----------------------------
BASELINE_ARGS_WINE: Dict[str, Any] = dict(
    dataset="wine",
    seed=42,
    train_fraction=0.8,
    split_strategy="balanced",
    n_micro=5,
    kmeans_n_init=50,
    kmeans_max_iter=500,
    dist_quantile=0.6,
    b_boundary=5,
    d_model=128,
    n_heads=2,
    attn_hidden=64,
    n_transformer_layers=3,
    train_epochs=2000,
    lr=0.003,
    weight_decay=0.001,
    k_target=3,
    data_root="datasets",
    out_dir="outputs/gothic/real",
)

# Confidence seeds (edit as you like)
DEFAULT_SEEDS: List[int] = [0, 1, 2, 3, 4, 42, 1337, 2024]

# One-factor ablation grid (seed fixed = 42). Edit freely.
ABLATIONS: Dict[str, List[Any]] = {
    "n_micro": [2, 3, 5, 8, 12, 20],
    "dist_quantile": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    "b_boundary": [1, 3, 5, 7, 10],
    "d_model": [32, 64, 128, 256],            # keep n_heads=2 baseline (divides all)
    "n_heads": [1, 2, 4, 8],                  # keep d_model=128 baseline (divisible)
    "attn_hidden": [32, 64, 128, 256],
    "n_transformer_layers": [1, 2, 3, 4, 6],
}

# Metrics we extract from *_results.txt
METRICS = ["test_acc", "test_nmi", "test_ari", "test_ami", "test_fmi"]


# -----------------------------
# Helpers
# -----------------------------
def safe_token(x: Any) -> str:
    """Make a filesystem/run_id-safe token for values like 0.6 -> 0p6."""
    s = str(x)
    s = s.replace(".", "p").replace("-", "m")
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s)
    return s


def dict_to_cli_args(d: Dict[str, Any]) -> List[str]:
    """Convert dict to CLI args: {'a':1,'b':2} -> ['--a','1','--b','2']"""
    out: List[str] = []
    for k, v in d.items():
        out.append(f"--{k}")
        out.append(str(v))
    return out


def parse_gothic_results_txt(path: Path) -> Dict[str, float]:
    """
    Parse out: Test ACC/NMI/ARI/AMI/FMI from <dataset>_results.txt
    Written by gothic_model.py.
    """
    txt = path.read_text(encoding="utf-8", errors="ignore")

    def grab(pattern: str) -> Optional[float]:
        m = re.search(pattern, txt)
        return float(m.group(1)) if m else None

    # Lines look like: "Test ACC         = 0.944444"
    parsed = {
        "test_nmi": grab(r"Test\s+NMI\s*=\s*([0-9.+-eE]+)"),
        "test_ami": grab(r"Test\s+AMI\s*=\s*([0-9.+-eE]+)"),
        "test_ari": grab(r"Test\s+ARI\s*=\s*([0-9.+-eE]+)"),
        "test_fmi": grab(r"Test\s+FMI\s*=\s*([0-9.+-eE]+)"),
        "test_acc": grab(r"Test\s+ACC\s*=\s*([0-9.+-eE]+)"),
    }

    missing = [k for k, v in parsed.items() if v is None]
    if missing:
        raise ValueError(f"Could not parse metrics {missing} from: {path}")

    return {k: float(v) for k, v in parsed.items()}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def stats_for_metric(values: List[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=float)
    return {
        "mean": float(arr.mean()) if len(arr) else float("nan"),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "var": float(arr.var(ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(arr.min()) if len(arr) else float("nan"),
        "max": float(arr.max()) if len(arr) else float("nan"),
        "n": int(len(arr)),
    }


def plot_seed_scatter(report_dir: Path, metric: str, seed_vals: List[Tuple[int, float]]) -> Path:
    """
    Scatter metric vs seed, with mean line.
    Transparent background.
    """
    seeds = [s for s, _ in seed_vals]
    vals = [v for _, v in seed_vals]
    m = float(np.mean(vals)) if vals else float("nan")

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.scatter(seeds, vals)
    ax.axhline(m, linewidth=1.5)
    ax.set_xlabel("seed")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} across seeds (mean line)")
    fig.tight_layout()

    out = report_dir / "plots" / f"seed_sweep_{metric}.png"
    ensure_dir(out.parent)
    fig.savefig(out, dpi=220, transparent=True)
    plt.close(fig)
    return out


def plot_seed_bar_meanstd(report_dir: Path, metric_stats: Dict[str, Dict[str, float]]) -> Path:
    """
    Bar plot of mean with error bars (std) for multiple metrics.
    Transparent background.
    """
    labels = list(metric_stats.keys())
    means = [metric_stats[k]["mean"] for k in labels]
    stds = [metric_stats[k]["std"] for k in labels]

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.bar(labels, means, yerr=stds, capsize=4)
    ax.set_ylabel("score")
    ax.set_title("Seed sweep: mean ± std (test metrics)")
    ax.set_xticklabels(labels, rotation=25, ha="right")
    fig.tight_layout()

    out = report_dir / "plots" / "seed_sweep_mean_std.png"
    ensure_dir(out.parent)
    fig.savefig(out, dpi=220, transparent=True)
    plt.close(fig)
    return out


def plot_ablation_curve(report_dir: Path, var: str, xs: List[Any], ys_acc: List[float], ys_nmi: List[float]) -> Path:
    """
    One plot per variable: ACC and NMI vs variable value.
    Transparent background.
    """
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(xs, ys_acc, marker="o", label="test_acc")
    ax.plot(xs, ys_nmi, marker="o", label="test_nmi")
    ax.set_xlabel(var)
    ax.set_ylabel("score")
    ax.set_title(f"Ablation: {var} (seed=42)")
    ax.legend()
    fig.tight_layout()

    out = report_dir / "plots" / f"ablation_{var}.png"
    ensure_dir(out.parent)
    fig.savefig(out, dpi=220, transparent=True)
    plt.close(fig)
    return out


# -----------------------------
# Runner
# -----------------------------
@dataclass
class RunResult:
    run_id: str
    dataset: str
    seed: int
    tag: str                  # "seed_sweep" or "ablation"
    varied_param: str         # "none" or param name
    varied_value: str         # "baseline" or value
    results_txt: str
    metrics: Dict[str, float]


def run_gothic_once(
    repo_root: Path,
    run_id: str,
    args: Dict[str, Any],
    skip_existing: bool = True,
) -> RunResult:
    """
    Executes:
      python runner.py --model gothic --run_id <run_id> ...args...
    Then parses out_dir/run_id/<dataset>_results.txt
    """
    dataset = str(args["dataset"])
    seed = int(args["seed"])
    out_dir = Path(str(args["out_dir"]))
    results_txt = out_dir / run_id / f"{dataset}_results.txt"

    if skip_existing and results_txt.exists():
        metrics = parse_gothic_results_txt(results_txt)
        return RunResult(
            run_id=run_id,
            dataset=dataset,
            seed=seed,
            tag=str(args.get("_tag", "unknown")),
            varied_param=str(args.get("_varied_param", "none")),
            varied_value=str(args.get("_varied_value", "baseline")),
            results_txt=str(results_txt),
            metrics=metrics,
        )

    cmd = [
        sys.executable,
        str(repo_root / "runner.py"),
        "--model", "gothic",
        "--run_id", run_id,
    ] + dict_to_cli_args({k: v for k, v in args.items() if not k.startswith("_") and k != "run_id"})

    print("\n[ABLATION] Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(repo_root), check=True)

    if not results_txt.exists():
        raise FileNotFoundError(f"Expected results file not found: {results_txt}")

    metrics = parse_gothic_results_txt(results_txt)

    return RunResult(
        run_id=run_id,
        dataset=dataset,
        seed=seed,
        tag=str(args.get("_tag", "unknown")),
        varied_param=str(args.get("_varied_param", "none")),
        varied_value=str(args.get("_varied_value", "baseline")),
        results_txt=str(results_txt),
        metrics=metrics,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=BASELINE_ARGS_WINE["dataset"])
    parser.add_argument("--out_dir", default=BASELINE_ARGS_WINE["out_dir"])
    parser.add_argument("--data_root", default=BASELINE_ARGS_WINE["data_root"])
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--no_skip_existing", action="store_true", help="Force reruns even if results exist.")
    parser.add_argument("--only_seed_sweep", action="store_true")
    parser.add_argument("--only_ablations", action="store_true")
    args = parser.parse_args()

    skip_existing = args.skip_existing and not args.no_skip_existing

    repo_root = Path(__file__).resolve().parent
    runner_path = repo_root / "runner.py"
    if not runner_path.exists():
        raise FileNotFoundError(f"runner.py not found next to this script: {runner_path}")

    # Build baseline from your example, but allow dataset/out_dir/data_root override
    baseline = dict(BASELINE_ARGS_WINE)
    baseline["dataset"] = args.dataset
    baseline["out_dir"] = args.out_dir
    baseline["data_root"] = args.data_root

    # Report folder
    ts = time.strftime("%Y%m%d_%H%M%S")
    report_dir = Path(baseline["out_dir"]) / "ABlationReports" / baseline["dataset"] / ts
    ensure_dir(report_dir)

    # Save the config used
    (report_dir / "config_used.json").write_text(json.dumps({
        "baseline": baseline,
        "seeds": DEFAULT_SEEDS,
        "ablations": ABLATIONS,
    }, indent=2), encoding="utf-8")

    all_runs: List[RunResult] = []

    # -----------------
    # 1) Seed sweep
    # -----------------
    if not args.only_ablations:
        print("\n========== Seed sweep (confidence) ==========")
        seed_rows: List[Dict[str, Any]] = []

        for s in DEFAULT_SEEDS:
            run_id = f"{baseline['dataset']}_seed{int(s):04d}_BASE"
            cfg = dict(baseline)
            cfg["seed"] = int(s)
            cfg["_tag"] = "seed_sweep"
            cfg["_varied_param"] = "seed"
            cfg["_varied_value"] = str(s)

            rr = run_gothic_once(repo_root, run_id=run_id, args=cfg, skip_existing=skip_existing)
            all_runs.append(rr)

            row = {
                "tag": rr.tag,
                "dataset": rr.dataset,
                "run_id": rr.run_id,
                "seed": rr.seed,
                **rr.metrics,
                "results_txt": rr.results_txt,
            }
            seed_rows.append(row)

        # Save seed sweep raw results
        seed_csv = report_dir / "seed_sweep_raw.csv"
        save_csv(seed_csv, seed_rows, fieldnames=["tag", "dataset", "run_id", "seed"] + METRICS + ["results_txt"])

        # Stats
        metric_stats: Dict[str, Dict[str, float]] = {}
        for m in METRICS:
            metric_stats[m] = stats_for_metric([float(r[m]) for r in seed_rows])

        stats_rows = []
        for m, st in metric_stats.items():
            stats_rows.append({"metric": m, **st})

        seed_stats_csv = report_dir / "seed_sweep_stats.csv"
        save_csv(seed_stats_csv, stats_rows, fieldnames=["metric", "mean", "std", "var", "min", "max", "n"])

        # Plots (transparent)
        # Scatter per metric (ACC + NMI by default; extend if you want)
        seed_vals_acc = [(int(r["seed"]), float(r["test_acc"])) for r in seed_rows]
        seed_vals_nmi = [(int(r["seed"]), float(r["test_nmi"])) for r in seed_rows]
        plot_seed_scatter(report_dir, "test_acc", seed_vals_acc)
        plot_seed_scatter(report_dir, "test_nmi", seed_vals_nmi)
        plot_seed_bar_meanstd(report_dir, metric_stats)

        print(f"[OK] Seed sweep saved:\n  {seed_csv}\n  {seed_stats_csv}\n  {report_dir / 'plots'}")

    # -----------------
    # 2) One-factor ablations (seed=42)
    # -----------------
    if not args.only_seed_sweep:
        print("\n========== One-factor ablations (seed=42) ==========")
        ablation_rows: List[Dict[str, Any]] = []

        fixed_seed = 42
        for var, values in ABLATIONS.items():
            xs = []
            ys_acc = []
            ys_nmi = []

            for v in values:
                cfg = dict(baseline)
                cfg["seed"] = fixed_seed
                cfg[var] = v
                cfg["_tag"] = "ablation"
                cfg["_varied_param"] = var
                cfg["_varied_value"] = str(v)

                # run_id: dataset_abl_<var>_<value>_seed42
                run_id = f"{baseline['dataset']}_abl_{var}_{safe_token(v)}_seed{fixed_seed}"

                rr = run_gothic_once(repo_root, run_id=run_id, args=cfg, skip_existing=skip_existing)
                all_runs.append(rr)

                row = {
                    "tag": rr.tag,
                    "dataset": rr.dataset,
                    "run_id": rr.run_id,
                    "seed": rr.seed,
                    "varied_param": rr.varied_param,
                    "varied_value": rr.varied_value,
                    **rr.metrics,
                    "results_txt": rr.results_txt,
                }
                ablation_rows.append(row)

                xs.append(v)
                ys_acc.append(rr.metrics["test_acc"])
                ys_nmi.append(rr.metrics["test_nmi"])

            # Plot ACC/NMI vs variable
            plot_ablation_curve(report_dir, var, xs, ys_acc, ys_nmi)

        # Save ablation raw results
        abl_csv = report_dir / "ablations_raw.csv"
        save_csv(
            abl_csv,
            ablation_rows,
            fieldnames=["tag", "dataset", "run_id", "seed", "varied_param", "varied_value"] + METRICS + ["results_txt"],
        )

        print(f"[OK] Ablations saved:\n  {abl_csv}\n  {report_dir / 'plots'}")

    # -----------------
    # 3) Save all runs combined
    # -----------------
    all_rows = []
    for rr in all_runs:
        all_rows.append({
            "tag": rr.tag,
            "dataset": rr.dataset,
            "run_id": rr.run_id,
            "seed": rr.seed,
            "varied_param": rr.varied_param,
            "varied_value": rr.varied_value,
            **rr.metrics,
            "results_txt": rr.results_txt,
        })

    combined_csv = report_dir / "all_runs_combined.csv"
    save_csv(
        combined_csv,
        all_rows,
        fieldnames=["tag", "dataset", "run_id", "seed", "varied_param", "varied_value"] + METRICS + ["results_txt"],
    )

    print("\n========== DONE ==========")
    print("Report folder:")
    print(f"  {report_dir}")
    print("Key outputs:")
    print(f"  {combined_csv}")
    print(f"  {report_dir / 'plots'}")


if __name__ == "__main__":
    main()
