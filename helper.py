"""
Helper utilities for GOTHIC: Graph-Overclustered Transformer-based Hierarchical Integrated Clustering.

Contains:
  - set_seed
  - majority_label
  - clustering_accuracy_with_map
  - apply_label_mapping
  - UnionFind
"""

import math
import random
from typing import Dict, Tuple

import numpy as np
import torch

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def majority_label(y_subset: np.ndarray):
    if len(y_subset) == 0:
        return None
    counts = np.bincount(y_subset.astype(int))
    return int(counts.argmax())


def clustering_accuracy_with_map(y_true, y_pred) -> Tuple[float, Dict[int, int]]:
    """
    Accuracy after optimal label permutation.
    Returns (acc, mapping_dict).

    y_true, y_pred: 1D integer arrays of the same shape.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    assert y_true.shape == y_pred.shape
    D = max(y_true.max(), y_pred.max()) + 1
    W = np.zeros((D, D), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        W[t, p] += 1

    if SCIPY_AVAILABLE:
        row_ind, col_ind = linear_sum_assignment(W.max() - W)
    else:
        # Greedy fallback
        row_ind = []
        col_ind = []
        W_copy = W.copy()
        for _ in range(D):
            i, j = divmod(W_copy.argmax(), D)
            if W_copy[i, j] == 0:
                break
            row_ind.append(i)
            col_ind.append(j)
            W_copy[i, :] = -1
            W_copy[:, j] = -1
        row_ind = np.array(row_ind, dtype=int)
        col_ind = np.array(col_ind, dtype=int)

    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    matched = sum(W[row, col] for row, col in zip(row_ind, col_ind))
    acc = matched / float(len(y_true))
    return acc, mapping


def apply_label_mapping(y_pred, mapping: Dict[int, int]):
    return np.array([mapping.get(c, c) for c in y_pred], dtype=int)


class UnionFind:
    """
    Simple union-find / disjoint set data structure.
    """

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.set_count = n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        self.set_count -= 1
        return True
