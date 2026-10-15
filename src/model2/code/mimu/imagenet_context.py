"""ImageNet V2 ResNet101 encoder with ZIP Context448 scSE decoder."""
import hashlib

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet101

from .resnet34_unet import ConvBlock, ResNet34UNet


class ImageNetContextUNet(ResNet34UNet):
    def __init__(self, weights_path, expected_sha256):
        nn.Module.__init__(self)
        digest = hashlib.sha256(weights_path.read_bytes()).hexdigest()
        if digest != expected_sha256:
            raise ValueError(f"ImageNet ResNet101 SHA-256 mismatch: {digest}")
        encoder = resnet101(weights=None)
        encoder.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True), strict=True)
        self.conv1, self.bn1, self.relu, self.maxpool = encoder.conv1, encoder.bn1, encoder.relu, encoder.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = encoder.layer1, encoder.layer2, encoder.layer3, encoder.layer4
        self.decoder4 = ConvBlock(2048 + 1024, 256, True)
        self.decoder3 = ConvBlock(256 + 512, 128, True)
        self.decoder2 = ConvBlock(128 + 256, 64, True)
        self.decoder1 = ConvBlock(64 + 64, 64, True)
        self.decoder0 = ConvBlock(64, 32, True)
        self.head = nn.Conv2d(32, 4, 1)
        self.register_buffer("input_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1))
        self.register_buffer("input_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1))
        self.pretrained = False  # RGB normalization below, not parent grayscale normalization.
        self.use_scse = True

    def forward(self, x, auxiliary=False):
        x = (x.repeat(1, 3, 1, 1) - self.input_mean) / self.input_std
        if getattr(self, "swin_encoder", False):
            input_size = x.shape[-2:]
            skips = []
            for index, layer in enumerate(self.encoder):
                x = layer(x)
                if index in (1, 3, 5, 7):
                    feature = self.encoder_norm(x) if index == 7 else x
                    skips.append(feature.permute(0, 3, 1, 2).contiguous())
            stem = F.interpolate(skips[0], size=(input_size[0] // 2, input_size[1] // 2), mode="bilinear", align_corners=False)
            x = skips[-1]
            for skip, decoder in ((skips[2], self.decoder4), (skips[1], self.decoder3), (skips[0], self.decoder2), (stem, self.decoder1)):
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
                x = decoder(torch.cat((x, skip), 1))
                if decoder is self.decoder4 and hasattr(self, "global_context"):
                    x = x + self.global_context(x)
            features = self.decoder0(F.interpolate(x, size=input_size, mode="bilinear", align_corners=False))
            logits = self.head(features)
        else:
            logits, features = super().forward(x, return_features=True)
        if self.training and not getattr(self, "full_context_supervision", False):
            assert logits.shape[-2:] == (448, 448)
            logits, features = logits[..., 112:336, 112:336], features[..., 112:336, 112:336]
        return (logits, features) if auxiliary else logits
