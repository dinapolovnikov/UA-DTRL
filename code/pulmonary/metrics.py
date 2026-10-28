import numpy as np
from numpy.typing import ArrayLike, NDArray


def _binary(labels: ArrayLike, scores: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    y = np.asarray(labels, dtype=float).reshape(-1)
    p = np.asarray(scores, dtype=float).reshape(-1)
    if y.size == 0 or y.shape != p.shape or not np.isin(y, [0, 1]).all():
        raise ValueError("labels and scores must be equal nonempty vectors with binary labels")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("scores must be finite probabilities")
    return y, p


def auroc(labels: ArrayLike, scores: ArrayLike) -> float:
    y, p = _binary(labels, scores)
    positive, negative = p[y == 1], p[y == 0]
    if not len(positive) or not len(negative):
        raise ValueError("AUROC requires both classes")
    order = np.argsort(p, kind="stable")
    ranks = np.empty(len(p), dtype=float)
    sorted_scores = p[order]
    start = 0
    while start < len(p):
        end = start + 1
        while end < len(p) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return float((ranks[y == 1].sum() - len(positive) * (len(positive) + 1) / 2) / (len(positive) * len(negative)))


def brier(labels: ArrayLike, scores: ArrayLike) -> float:
    y, p = _binary(labels, scores)
    return float(np.mean((y - p) ** 2))


def calibration_error(labels: ArrayLike, scores: ArrayLike, bins: int = 10) -> tuple[float, float]:
    y, p = _binary(labels, scores)
    if bins < 1:
        raise ValueError("bins must be positive")
    groups = np.array_split(np.argsort(p, kind="stable"), min(bins, len(y)))
    errors = np.array([abs(y[g].mean() - p[g].mean()) for g in groups])
    sizes = np.array([len(g) for g in groups])
    return float(np.dot(errors, sizes) / len(y)), float(errors.max())


def net_benefit(labels: ArrayLike, scores: ArrayLike, cutoff: float = 0.16) -> float:
    y, p = _binary(labels, scores)
    if not 0 < cutoff < 1:
        raise ValueError("cutoff must be in (0, 1)")
    selected = p >= cutoff
    tp = np.sum(selected & (y == 1))
    fp = np.sum(selected & (y == 0))
    return float((tp - fp * cutoff / (1 - cutoff)) / len(y))
