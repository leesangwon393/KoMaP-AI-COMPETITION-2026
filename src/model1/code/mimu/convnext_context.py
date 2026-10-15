"""ConvNeXt-Tiny ImageNet encoder with ZIP 448→center224 scSE decoder."""
import hashlib

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import convnext_tiny

from .resnet34_unet import ConvBlock
from .detail_residual import ResidualDetailRefiner


class ConvNeXtContextUNet(nn.Module):
    def __init__(self, weights_path, expected_sha256):
        super().__init__()
        digest = hashlib.sha256(weights_path.read_bytes()).hexdigest()
        if digest != expected_sha256:
            raise ValueError(f"ConvNeXt-Tiny SHA-256 mismatch: {digest}")
        backbone = convnext_tiny(weights=None)
        backbone.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True), strict=True)
        self.encoder = backbone.features
        self.decoder4 = ConvBlock(768 + 384, 256, True)
        self.decoder3 = ConvBlock(256 + 192, 128, True)
        self.decoder2 = ConvBlock(128 + 96, 64, True)
        self.detail = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.decoder1 = ConvBlock(64 + 16, 64, True)
        self.decoder0 = ConvBlock(64 + 1, 32, True)
        self.head = nn.Conv2d(32, 4, 1)
        self.register_buffer("input_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1))
        self.register_buffer("input_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1))

        with torch.random.fork_rng(devices=[]):
            self.detail_refiner = ResidualDetailRefiner()

    def forward(self, x, auxiliary=False):
        raw = x
        resize_mode = getattr(self, 'decoder_resize_mode', 'bilinear')
        align_corners = False if resize_mode == 'bilinear' else None
        normalized = (x.repeat(1, 3, 1, 1) - self.input_mean) / self.input_std
        skips = []
        for index, layer in enumerate(self.encoder):
            normalized = layer(normalized)
            if index in (1, 3, 5, 7):
                feature = normalized
                skips.append(feature)
        decoded = skips[-1]
        for skip, block in zip(reversed(skips[:-1]), (self.decoder4, self.decoder3, self.decoder2)):
            decoded = F.interpolate(decoded, size=skip.shape[-2:], mode=resize_mode, align_corners=align_corners)
            decoded = block(torch.cat((decoded, skip), 1))
            if block is self.decoder4 and hasattr(self, 'global_context'):
                context = self.global_context(decoded)
                if context.shape[-2:] != (1, 1):
                    context = F.interpolate(context, size=decoded.shape[-2:], mode='bilinear', align_corners=False)
                decoded = decoded + context
        context_features = decoded
        detail = self.detail_refiner(self.detail(raw))
        decoded = self.decoder1(torch.cat((F.interpolate(decoded, size=detail.shape[-2:], mode=resize_mode, align_corners=align_corners), detail), 1))
        decoded = F.interpolate(decoded, size=raw.shape[-2:], mode=resize_mode, align_corners=align_corners)
        features = self.decoder0(torch.cat((decoded, raw), 1))
        logits = self.head(features)
        if self.training and not getattr(self, "full_context_supervision", False):
            assert raw.shape[-2:] == (448, 448)
            logits, features = logits[..., 112:336, 112:336], features[..., 112:336, 112:336]
            if auxiliary and hasattr(self, 'deep_supervision_head'):
                features = context_features
        return (logits, features) if auxiliary else logits
