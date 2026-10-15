"""Semantic-gated raw-detail residual correction for R073.

This is an independently written research adaptation; it does not import
Gated-SCNN code or weights.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .detail_residual import ResidualDetailRefiner


class SemanticGate(nn.Module):
    def __init__(self, semantic_channels, detail_channels):
        super().__init__()
        self.projection = nn.Conv2d(semantic_channels, detail_channels, 1)

    def forward(self, detail, semantic):
        context = F.interpolate(semantic, size=detail.shape[-2:], mode="bilinear", align_corners=False)
        gate_logits = self.projection(context)
        return detail * torch.sigmoid(gate_logits)


class SemanticGatedDetailUNet(nn.Module):
    """Wrap a fresh semantic segmenter with a gated detail path."""

    def __init__(self, semantic_model, detail_channels=16, detail_stride=2):
        super().__init__()
        self.semantic = semantic_model
        self.detail_stride = detail_stride
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(73073)
            self.detail_stem = nn.Conv2d(1, detail_channels, 3, stride=detail_stride, padding=1, bias=False)
            self.detail_refiner = ResidualDetailRefiner()
            self.semantic_gate = SemanticGate(32, detail_channels)
            self.detail_to_semantic = nn.Conv2d(detail_channels, 32, 1, bias=False)
            self.residual_fusion = nn.Conv2d(32, 32, 1, bias=False)
            nn.init.zeros_(self.residual_fusion.weight)

    def forward(self, image, auxiliary=False):
        _, semantic_features = self.semantic(image, auxiliary=True)
        detail_input = image
        detail = self.detail_stem(detail_input)
        detail = self.detail_refiner(detail)
        if self.training and not getattr(self, "full_context_supervision", False):
            if image.shape[-2:] != (448, 448):
                raise ValueError("training requires native 448x448 context input")
            start, stop = 112 // self.detail_stride, 336 // self.detail_stride
            detail = detail[..., start:stop, start:stop]
        gated_detail = self.semantic_gate(detail, semantic_features)
        correction = self.detail_to_semantic(gated_detail)
        correction = F.interpolate(
            correction, size=semantic_features.shape[-2:], mode="bilinear", align_corners=False
        )
        features = semantic_features + self.residual_fusion(correction)
        logits = self.semantic.head(features)
        return (logits, features) if auxiliary else logits
