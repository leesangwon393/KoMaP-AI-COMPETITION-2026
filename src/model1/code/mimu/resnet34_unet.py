"""Shared convolution/scSE blocks and the semantic U-Net forward path."""
import torch
from torch import nn
from torch.nn import functional as F


class SCSE(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1), nn.Sigmoid())
        self.spatial = nn.Sequential(nn.Conv2d(channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.channel(x) + x * self.spatial(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_scse):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.attention = SCSE(out_channels) if use_scse else nn.Identity()

    def forward(self, x):
        features = self.attention(self.block(x))
        if hasattr(self, "shortcut_projection"):
            features = features + self.shortcut_projection(x)
        return features
