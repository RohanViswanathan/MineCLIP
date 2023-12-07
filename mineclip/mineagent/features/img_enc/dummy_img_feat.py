"""
Note that image feature is provided by MineCLIP.
"""
import torch
import torch.nn as nn

class DummyImgFeat(nn.Module):
    def __init__(self, device: torch.device, output_dim: int = 512):
        super().__init__()
        self._device = device
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 40 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        conv_out = self.conv_layers(x)
        flatten = conv_out.view(conv_out.size(0), -1)  # Flatten the output for FC layers
        output = self.fc_layers(flatten)
        return output.to(self._device), None

# class DummyImgFeat(nn.Module):
#     def __init__(self, *, output_dim: int = 512, device: torch.device):
#         super().__init__()
#         self._output_dim = output_dim
#         self._device = device

#     @property
#     def output_dim(self):
#         return self._output_dim

#     def forward(self, x, **kwargs):
#         return x.to(self._device), None
