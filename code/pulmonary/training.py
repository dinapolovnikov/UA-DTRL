import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from .evidence import nig_loss
from .model import EvidentialCritic, PulmonaryEncoder


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def critic_step(encoder: PulmonaryEncoder, critic: EvidentialCritic, optimizer: torch.optim.Optimizer,
                batch: dict[str, Tensor], regularizer_weight: float) -> float:
    encoder.train()
    critic.train()
    optimizer.zero_grad(set_to_none=True)
    state = encoder(batch["physiology"], batch["image"], batch["image_present"])
    loss = nig_loss(critic(state, batch["action"]), batch["target"], regularizer_weight)
    if not torch.isfinite(loss):
        raise ValueError("non-finite loss")
    torch.autograd.backward(loss)
    optimizer.step()
    return float(loss.detach())


def save_state(path: str | Path, encoder: nn.Module, critic: nn.Module,
               optimizer: torch.optim.Optimizer, seed: int) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="state-", suffix=".pt", dir=destination.parent)
    os.close(descriptor)
    try:
        torch.save({"encoder": encoder.state_dict(), "critic": critic.state_dict(),
                    "optimizer": optimizer.state_dict(), "seed": seed,
                    "torch_rng": torch.get_rng_state()}, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_state(path: str | Path, encoder: nn.Module, critic: nn.Module,
               optimizer: torch.optim.Optimizer) -> int:
    state = torch.load(path, map_location="cpu", weights_only=True)
    encoder.load_state_dict(state["encoder"])
    critic.load_state_dict(state["critic"])
    optimizer.load_state_dict(state["optimizer"])
    set_seed(int(state["seed"]))
    torch.set_rng_state(state["torch_rng"])
    return int(state["seed"])
