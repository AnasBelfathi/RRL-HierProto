# aggregators.py
# ---------------------------------------------------------------------
import numpy as np
from typing import List


def mean(vectors: List[np.ndarray]) -> np.ndarray:
    return np.mean(vectors, axis=0)


def median(vectors: List[np.ndarray]) -> np.ndarray:
    return np.median(vectors, axis=0)


STRATEGIES = {
    "mean": mean,
    "median": median,
    # on pourra en ajouter (« max », « attention », …)
}
