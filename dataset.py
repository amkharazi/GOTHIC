"""
dataset.py

Dataset downloader/generator for GOTHIC.
Supports:
  Synthetic (SIPU-like 2D clusters, stored in datasets/synthetic):
    - compound
    - aggregation
    - d31
    - flame
    - jain
    - pathbased
    - r15
    - noisy_circles (generated via sklearn)

  Real datasets (stored in datasets/real):
    - breast_cancer
    - iris
    - wine

Usage:
  python dataset.py                 # download/generate all datasets
  python dataset.py --datasets compound aggregation noisy_circles
"""

import sys
from pathlib import Path
from typing import List

import numpy as np

# Ensure local imports work when run as a script
sys.path.append(str(Path(__file__).resolve().parent))

from sklearn.datasets import make_circles, load_breast_cancer, load_iris, load_wine

SIPU_BASE_URL = "http://cs.joensuu.fi/sipu/datasets/"

SYNTHETIC_DIR = Path("datasets") / "synthetic"
REAL_DIR = Path("datasets") / "real"

SIPU_DATASETS = {
    "compound": "Compound.txt",
    "aggregation": "Aggregation.txt",
    "d31": "D31.txt",
    "flame": "flame.txt",
    "jain": "jain.txt",
    "pathbased": "pathbased.txt",
    "r15": "R15.txt",
}

REAL_DATASETS = {"breast_cancer", "iris", "wine"}

SUPPORTED_DATASETS = set(SIPU_DATASETS.keys()) | {"noisy_circles"} | REAL_DATASETS


def download_sipu_dataset(name: str, base_dir: Path = Path("datasets")) -> Path:
    """
    Download a SIPU dataset (txt) into datasets/synthetic.
    """
    name = name.lower()
    if name not in SIPU_DATASETS:
        raise ValueError(f"Unknown SIPU dataset: {name}")

    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    filename = SIPU_DATASETS[name]
    out_path = SYNTHETIC_DIR / filename

    if out_path.exists():
        print(f"[dataset] {name} already present at {out_path}")
        return out_path

    import urllib.request

    url = SIPU_BASE_URL + filename
    print(f"[dataset] Downloading {name} from {url}")
    urllib.request.urlretrieve(url, out_path.as_posix())
    print(f"[dataset] Saved {name} to {out_path}")
    return out_path


def generate_noisy_circles(base_dir: Path = Path("datasets")) -> Path:
    """
    Generate a 'noisy circles' dataset via sklearn and save as CSV.
    This avoids needing Kaggle credentials.
    """
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SYNTHETIC_DIR / "noisy_circles.csv"

    if out_path.exists():
        print(f"[dataset] noisy_circles already present at {out_path}")
        return out_path

    print("[dataset] Generating noisy_circles via sklearn.make_circles ...")
    X, y = make_circles(n_samples=1500, factor=0.5, noise=0.05, random_state=42)
    data = np.column_stack([X, y])
    header = "x1,x2,label"
    np.savetxt(out_path, data, delimiter=",", header=header, comments="")
    print(f"[dataset] Saved noisy_circles to {out_path}")
    return out_path


def generate_real_dataset(name: str, base_dir: Path = Path("datasets")) -> Path:
    """
    Use sklearn to materialize real datasets as CSV:
      breast_cancer.csv, iris.csv, wine.csv in datasets/real.
    """
    name = name.lower()
    if name not in REAL_DATASETS:
        raise ValueError(f"Unknown real dataset: {name}")

    REAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REAL_DIR / f"{name}.csv"

    if out_path.exists():
        print(f"[dataset] {name} already present at {out_path}")
        return out_path

    if name == "breast_cancer":
        ds = load_breast_cancer()
    elif name == "iris":
        ds = load_iris()
    elif name == "wine":
        ds = load_wine()
    else:
        raise ValueError(f"Unsupported real dataset: {name}")

    X = ds.data
    y = ds.target
    data = np.column_stack([X, y])
    # Feature names + label
    feature_names = ds.feature_names if hasattr(ds, "feature_names") else [f"x{i}" for i in range(X.shape[1])]
    header = ",".join(list(feature_names) + ["label"])
    np.savetxt(out_path, data, delimiter=",", header=header, comments="")
    print(f"[dataset] Saved {name} to {out_path}")
    return out_path


def ensure_dataset_exists(name: str, base_dir: Path = Path("datasets")) -> Path:
    """
    Ensure that the given dataset has been downloaded/generated.
    Returns the path to the underlying file.
    """
    name = name.lower()
    if name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Supported: {sorted(SUPPORTED_DATASETS)}")

    if name in SIPU_DATASETS:
        return download_sipu_dataset(name, base_dir=base_dir)
    elif name == "noisy_circles":
        return generate_noisy_circles(base_dir=base_dir)
    elif name in REAL_DATASETS:
        return generate_real_dataset(name, base_dir=base_dir)
    else:
        raise ValueError(f"Unhandled dataset: {name}")


def download_many(datasets: List[str]):
    """
    Download/generate multiple datasets by name.
    """
    for ds in datasets:
        ensure_dataset_exists(ds)


def parse_args(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Download/generate datasets for GOTHIC.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=[],
        help="Dataset names to download. If empty, download all supported datasets.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.datasets:
        targets = sorted(SUPPORTED_DATASETS)
        print("[dataset] No datasets specified; downloading all:")
        print("          " + ", ".join(targets))
    else:
        targets = [ds.lower() for ds in args.datasets]
        unknown = [t for t in targets if t not in SUPPORTED_DATASETS]
        if unknown:
            raise ValueError(f"Unknown dataset(s): {unknown}. Supported: {sorted(SUPPORTED_DATASETS)}")

    download_many(targets)


if __name__ == "__main__":
    main()
