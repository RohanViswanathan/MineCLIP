import torch
import torch.nn as nn

import mineclip.utils as U


class PrevActionEmb(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        output_dim: int,
        device: torch.device,
    ):
        super().__init__()
        self._embed = nn.Embedding(89, embed_dim)
        self._output_dim = output_dim
        self._device = device

    @property
    def output_dim(self):
        return self._output_dim

    def forward(self, x, **kwargs):
        x = U.any_to_torch_tensor(x, device=self._device)
        x = self._embed(x)
        x = torch.t(x)
        return x, None