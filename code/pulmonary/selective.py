import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


def threshold(calibration: ArrayLike, alpha: float = 0.1) -> float:
    values = np.asarray(calibration, dtype=float).reshape(-1)
    if not 0 < alpha < 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("finite nonempty calibration values and alpha in (0, 1) required")
    k = math.ceil((values.size + 1) * (1 - alpha))
    if k > values.size:
        return math.inf
    return float(np.partition(values, k - 1)[k - 1])


def defer(total_uncertainty: ArrayLike, cutoff: float) -> NDArray[np.bool_]:
    values = np.asarray(total_uncertainty, dtype=float)
    if not np.isfinite(values).all() or math.isnan(cutoff):
        raise ValueError("uncertainties must be finite and cutoff must not be NaN")
    return values > cutoff
