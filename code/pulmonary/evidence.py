import math

import torch
from torch import Tensor

from .model import Evidence


def uncertainty(evidence: Evidence) -> tuple[Tensor, Tensor]:
    aleatoric = evidence.beta / (evidence.alpha - 1)
    epistemic = aleatoric / evidence.nu
    return epistemic, aleatoric


def nig_loss(evidence: Evidence, target: Tensor, regularizer_weight: float) -> Tensor:
    mean, nu, alpha, beta = evidence
    error = target - mean
    omega = 2 * beta * (1 + nu)
    nll = (0.5 * torch.log(math.pi / nu) - alpha * torch.log(omega)
           + (alpha + 0.5) * torch.log(nu * error.square() + omega)
           + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5))
    penalty = error.abs() * (2 * nu + alpha)
    return (nll + regularizer_weight * penalty).mean()
