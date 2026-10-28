from typing import NamedTuple, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class Evidence(NamedTuple):
    mean: Tensor
    nu: Tensor
    alpha: Tensor
    beta: Tensor


class PulmonaryEncoder(nn.Module):
    def __init__(self, physiological_features: int, width: int = 128) -> None:
        super().__init__()
        if width % 4:
            raise ValueError("width must be divisible by four")
        self.width = width
        self.image = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, width, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.physiology = nn.Linear(physiological_features, width)
        self.position = nn.Parameter(torch.zeros(1, 48, width))
        block = nn.TransformerEncoderLayer(width, 4, 4 * width, dropout=0.0, batch_first=True)
        self.temporal = nn.TransformerEncoder(block, 2, enable_nested_tensor=False)
        self.attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, width))
        self.missing_gate = nn.Parameter(torch.zeros(()))
        self.normalization = nn.LayerNorm(width)

    def forward(self, physiology: Tensor, image: Tensor, image_present: Tensor) -> Tensor:
        if physiology.ndim != 3 or physiology.shape[1] != 48:
            raise ValueError("physiology must have shape [batch, 48, features]")
        if image.shape != (physiology.shape[0], 1, 32, 32):
            raise ValueError("image must have shape [batch, 1, 32, 32]")
        if image_present.shape != (physiology.shape[0],):
            raise ValueError("image_present must have shape [batch]")
        sequence = self.temporal(self.physiology(physiology) + self.position)
        encoded_image = self.image(image).unsqueeze(1)
        present = image_present.bool().view(-1, 1, 1)
        query = torch.where(present, encoded_image, self.mask_token.expand_as(encoded_image))
        context, _ = self.attention(query, sequence, sequence, need_weights=False)
        scale = torch.where(present, torch.ones_like(query), torch.sigmoid(self.missing_gate))
        return cast(Tensor, self.normalization(query + scale * context).squeeze(1))


def mlp(input_width: int, output_width: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_width, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, output_width),
    )


class EvidentialCritic(nn.Module):
    def __init__(self, state_width: int, action_width: int = 3) -> None:
        super().__init__()
        self.network = mlp(state_width + action_width, 4)

    def forward(self, state: Tensor, action: Tensor) -> Evidence:
        mean, nu, alpha, beta = self.network(torch.cat((state, action), dim=-1)).unbind(-1)
        return Evidence(mean, F.softplus(nu) + 0.1, F.softplus(alpha) + 1.1, F.softplus(beta) + 1e-6)


class GaussianActor(nn.Module):
    def __init__(self, state_width: int, action_width: int = 3) -> None:
        super().__init__()
        self.network = mlp(state_width, action_width * 2)

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor]:
        mean, log_scale = self.network(state).chunk(2, dim=-1)
        return mean, log_scale.clamp(-5, 2).exp()
