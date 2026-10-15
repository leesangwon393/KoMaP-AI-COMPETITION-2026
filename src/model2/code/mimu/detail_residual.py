"""Fixed 16-channel residual detail refiner; zero fusion starts as identity."""
import torch
from torch import nn


class ResidualDetailRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.dw3 = nn.Conv2d(16, 16, 3, padding=1, groups=16, bias=False)
        self.dw5 = nn.Conv2d(16, 16, 5, padding=2, groups=16, bias=False)
        self.activation = nn.ReLU()
        self.fusion = nn.Conv2d(32, 16, 1, bias=True)
        nn.init.zeros_(self.fusion.weight)
        nn.init.zeros_(self.fusion.bias)

    def forward(self, detail):
        branches = torch.cat((self.activation(self.dw3(detail)),
                              self.activation(self.dw5(detail))), dim=1)
        return detail + self.fusion(branches)
