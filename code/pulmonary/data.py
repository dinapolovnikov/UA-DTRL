from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TransitionDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, path: str | Path) -> None:
        with np.load(path, allow_pickle=False) as archive:
            required = ("physiology", "image", "image_present", "action", "target")
            missing = set(required).difference(archive.files)
            if missing:
                raise ValueError(f"missing arrays: {sorted(missing)}")
            self.arrays = {key: np.asarray(archive[key], dtype=np.float32) for key in required}
        n = self.arrays["target"].shape[0]
        if n == 0 or any(a.shape[0] != n for a in self.arrays.values()):
            raise ValueError("array lengths must agree and be nonzero")
        if self.arrays["physiology"].ndim != 3 or self.arrays["physiology"].shape[1] != 48:
            raise ValueError("physiology must have shape [N, 48, features]")
        if self.arrays["image"].shape != (n, 1, 32, 32):
            raise ValueError("image must have shape [N, 1, 32, 32]")
        if self.arrays["action"].shape != (n, 3) or self.arrays["target"].shape != (n,):
            raise ValueError("action must be [N, 3] and target must be [N]")
        if self.arrays["image_present"].shape != (n,) or not np.isin(self.arrays["image_present"], [0, 1]).all():
            raise ValueError("image_present must be a binary vector")
        if any(not np.isfinite(a).all() for a in self.arrays.values()):
            raise ValueError("arrays must be finite")

    def __len__(self) -> int:
        return len(self.arrays["target"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {name: torch.from_numpy(np.array(array[index], copy=True)) for name, array in self.arrays.items()}
