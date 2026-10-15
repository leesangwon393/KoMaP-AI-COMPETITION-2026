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
        return features + self.shortcut_projection(x) if hasattr(self, "shortcut_projection") else features


class ResNet34UNet(nn.Module):
    """Forward path initialized by the ImageNetContextUNet subclass."""

    def forward(self, x, return_features=False):
        input_size = x.shape[-2:]
        if self.pretrained:
            x = (x - self.input_mean) / self.input_std
        stem = self.relu(self.bn1(self.conv1(x)))
        e1 = self.layer1(self.maxpool(stem))
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        x = self.layer4(e3)
        for skip, decoder in ((e3, self.decoder4), (e2, self.decoder3),
                              (e1, self.decoder2), (stem, self.decoder1)):
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = decoder(torch.cat((x, skip), dim=1))
            if decoder is self.decoder4 and hasattr(self, 'global_context'):
                x = x + self.global_context(x)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        features = self.decoder0(x)
        logits = self.head(features)
        return (logits, features) if return_features else logits
