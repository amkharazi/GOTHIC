"""
loader.py

Dataset loader for GOTHIC.

Given a dataset name, it makes sure the data is present (by calling dataset.ensure_dataset_exists)
and then loads it into (X, y) NumPy arrays.

Conventions:
  - Synthetic SIPU-style txt datasets have 3 columns: x, y, label (1..K). We convert labels to 0..K-1.
  - noisy_circles.csv has columns: x1, x2, label (0..1).
  - Real CIF-like CSVs from sklearn have many feature columns and a final "label" column.
"""

import sys
from pathlib import Path
from typing import Tuple

import numpy as np

# Ensure local imports work when run as a script
sys.path.append(str(Path(__file__).resolve().parent))

from dataset import (
    ensure_dataset_exists,
    SIPU_DATASETS,
    REAL_DIR,
    SYNTHETIC_DIR,
    REAL_DATASETS,
)


def load_dataset(name: str, base_dir: str = "datasets") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset by name, returning (X, y).

    name: one of synthetic or real dataset names supported in dataset.py.
    """
    name = name.lower()
    base_dir_path = Path(base_dir)

    # Ensure dataset is present
    ensure_dataset_exists(name, base_dir=base_dir_path)

    if name in SIPU_DATASETS:
        filename = SIPU_DATASETS[name]
        path = SYNTHETIC_DIR / filename
        data = np.loadtxt(path)
        X = data[:, :2]
        y = data[:, 2].astype(int) - 1  # SIPU labels are 1..K
        return X, y

    if name == "noisy_circles":
        path = SYNTHETIC_DIR / "noisy_circles.csv"
        data = np.loadtxt(path, delimiter=",", skiprows=1)
        X = data[:, :2]
        y = data[:, 2].astype(int)
        return X, y

    if name in REAL_DATASETS:
        path = REAL_DIR / f"{name}.csv"
        data = np.loadtxt(path, delimiter=",", skiprows=1)
        X = data[:, :-1]
        y = data[:, -1].astype(int)
        return X, y

    raise ValueError(f"Unknown dataset: {name}")
