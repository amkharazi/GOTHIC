"""
runner.py

Generic runner for clustering models.

Usage examples:

  # Run GOTHIC on Compound
  python runner.py --model gothic --run_id ID001 --dataset compound --k_target 6 --n_micro 80

Any arguments after `--model ...` are forwarded to the corresponding model script.

Currently supported:
  --model gothic     -> gothic_model.py
  --model kmeans     -> kmeans_model.py
  --model dbscan     -> dbscan_model.py
  --model hdbscan    -> hdbscan_model.py
  --model insdpc     -> insdpc_model.py
  --model amd_dbscan -> amd_dbscan_model.py
  --model mdbscan    -> mdbscan_model.py
"""

import sys
from pathlib import Path
import subprocess
import argparse


def main():
    root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--model",
        type=str,
        default="gothic",
        help="Which model to run (e.g., 'gothic')."
    )

    # Parse only --model, leave the rest to be forwarded
    args, remaining = parser.parse_known_args()

    model = args.model.lower()

    if model == "gothic":
        script = root / "gothic_model.py"
    elif model == "kmeans":
        script = root / "kmeans_model.py"
    elif model == "dbscan":
        script = root / "dbscan_model.py"
    elif model == "hdbscan":
        script = root / "hdbscan_model.py"
    elif model == "insdpc":
        script = root / "insdpc_model.py"
    elif model == "amd_dbscan":
        script = root / "amd_dbscan_model.py"
    elif model == "mdbscan":
        script = root / "mdbscan_model.py"
    else:
        raise ValueError(f"Unknown model '{model}'. Currently supported: gothic , kmeans, dbscan, hdbscan, insdpc, amd_dbscan, mdbscan ")

    cmd = [sys.executable, str(script)] + remaining

    print("[RUNNER] Selected model:", model)
    print("[RUNNER] Executing command:")
    print("         " + " ".join(cmd))

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
