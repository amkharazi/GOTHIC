"""
runner.py

Generic runner for clustering models.

Usage examples:

  # Run GOTHIC on Compound
  python runner.py --model gothic --run_id ID001 --dataset compound --k_target 6 --n_micro 80

Any arguments after `--model ...` are forwarded to the corresponding model script.

Supported models:
  --model gothic          -> gothic_model.py
  --model kmeans          -> kmeans_model.py
  --model dbscan          -> dbscan_model.py
  --model hdbscan         -> hdbscan_model.py
  --model insdpc          -> insdpc_model.py
  --model amd_dbscan      -> amd_dbscan_model.py
  --model mdbscan         -> mdbscan_model.py

  --model spectral        -> spectral_model.py
  --model gnn             -> gnn_model.py
  --model idec            -> idec_model.py
  --model gnn_contrastive -> gnn_contrastive_model.py
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

    model_map = {
        "gothic": "gothic_model.py",
        "kmeans": "kmeans_model.py",
        "dbscan": "dbscan_model.py",
        "hdbscan": "hdbscan_model.py",
        "insdpc": "insdpc_model.py",
        "amd_dbscan": "amd_dbscan_model.py",
        "mdbscan": "mdbscan_model.py",
        "spectral": "spectral_model.py",
        "gnn": "gnn_model.py",
        "idec": "idec_model.py",
        "gnn_contrastive": "gnn_contrastive_model.py",
    }

    if model not in model_map:
        supported = ", ".join(sorted(model_map.keys()))
        raise ValueError(f"Unknown model '{model}'. Currently supported: {supported}")

    script = root / model_map[model]
    if not script.exists():
        raise FileNotFoundError(f"Model script not found: {script}")

    cmd = [sys.executable, str(script)] + remaining

    print("[RUNNER] Selected model:", model)
    print("[RUNNER] Executing command:")
    print("         " + " ".join(cmd))

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
