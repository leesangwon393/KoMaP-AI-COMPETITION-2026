"""Develop R052 and R073 independently, sequentially on pro6000 GPU0."""


from datetime import datetime


from functools import partial


import hashlib


import json


import multiprocessing


from concurrent.futures import ProcessPoolExecutor


from pathlib import Path


import shutil


import sys


from types import SimpleNamespace


from zoneinfo import ZoneInfo


sys.dont_write_bytecode = True


import numpy as np


from PIL import Image


import torch


class MultiScaleContext(torch.nn.Module):
    """Residual decoder context using three depthwise receptive fields."""

    def __init__(self, channels):
        super().__init__()
        self.paths = torch.nn.ModuleList([
            torch.nn.Conv2d(channels, channels, 3, padding=d, dilation=d,
                            groups=channels, bias=False) for d in (1, 2, 4)
        ])
        self.fusion = torch.nn.Conv2d(channels * 3, channels, 1, bias=False)
        torch.nn.init.zeros_(self.fusion.weight)

    def forward(self, features):
        context = torch.cat([torch.nn.functional.gelu(path(features)) for path in self.paths], 1)
        return features + self.fusion(context)


def regional_contrast(model, features, logits, target, generator):
    """ReCo-inspired supervised auxiliary; sparse queries, detached batch keys.

    Include boundary/thin-phase pixels. No memory bank, teacher or test data.
    Sampling uses checkpointed CPU auxiliary RNG, leaving crop RNG unchanged.
    """
    from torch.nn import functional as F
    assert features.shape[-2:] == target.shape[-2:] == logits.shape[-2:]
    vectors = features.permute(0, 2, 3, 1).reshape(-1, 32)
    truth = target.flatten().detach().cpu()
    confidence = logits.detach().softmax(1).gather(1, target[:, None]).flatten().cpu()
    pools = [(truth == cls).nonzero().flatten() for cls in range(4)]
    present = [cls for cls in range(4) if len(pools[cls])]
    if len(present) < 2:
        return features.sum() * 0
    with torch.no_grad():
        keys = F.normalize(model.projection(vectors), dim=1)
        prototypes = {cls: F.normalize(keys[pools[cls].to(features.device)].mean(0), dim=0)
                      for cls in present}
    losses = []
    for cls in present:
        hard = pools[cls][confidence[pools[cls]] <= .97]
        if not len(hard):
            continue
        selected = hard[torch.randperm(len(hard), generator=generator)[:32]]
        queries = F.normalize(model.projection(vectors[selected.to(features.device)]), dim=1)
        others = [other for other in present if other != cls]
        relations = torch.stack([prototypes[cls] @ prototypes[other] for other in others])
        classes = torch.multinomial(relations.cpu().softmax(0), 128, replacement=True,
                                    generator=generator)
        negatives = []
        for index in classes.tolist():
            pool = pools[others[index]]
            chosen = torch.randint(len(pool), (1,), generator=generator)
            negatives.append(pool[chosen])
        negative_keys = keys[torch.cat(negatives).to(features.device)]
        positive = (queries @ prototypes[cls])[:, None]
        similarities = torch.cat((positive, queries @ negative_keys.T), 1) / .1
        losses.append(F.cross_entropy(similarities, torch.zeros(len(queries), device=features.device,
                                                               dtype=torch.long)))
    return torch.stack(losses).mean() if losses else features.sum() * 0


def deep_supervision(model, features, target):
    from torch.nn import functional as F
    logits = F.interpolate(model.deep_supervision_head(features), size=(448, 448),
                           mode='bilinear', align_corners=False)[..., 112:336, 112:336]
    return F.cross_entropy(logits, target)


def feature_consistency(model, features, logits, generator):
    """Train-only channel dropout, with detached current prediction as target."""
    from torch.nn import functional as F
    keep = (torch.rand(features.shape[:2], generator=generator) >= .2).to(features.device)
    perturbed = features * keep[:, :, None, None] / .8
    prediction = model.semantic.head(perturbed)
    return F.kl_div(prediction.log_softmax(1), logits.detach().softmax(1),
                    reduction='none').sum(1).mean()


def phase_pair_supervision(logits, target):
    from torch.nn import functional as F
    pair = logits[:, 1:3].permute(0, 2, 3, 1)
    losses = []
    for cls in (1, 2):
        selected = target == cls
        if selected.any():
            losses.append(F.cross_entropy(pair[selected], torch.full((int(selected.sum()),), cls - 1,
                                          device=logits.device, dtype=torch.long)))
    return torch.stack(losses).mean() if losses else logits.sum() * 0


def geometric_consistency(model, images, logits, generator):
    """Train-only rotation equivariance, with the current prediction detached."""
    from torch.nn import functional as F
    rotation = int(torch.randint(1, 4, (), generator=generator))
    prediction = model(images.rot90(rotation, (-2, -1))).rot90(-rotation, (-2, -1))
    return F.kl_div(prediction.log_softmax(1), logits.detach().softmax(1),
                    reduction='none').sum(1).mean()


def context_cutmix(images, labels, generator):
    assert images.shape[-2:] == (448, 448) and labels.shape[-2:] == (224, 224)
    if torch.rand((), generator=generator).item() >= .5:
        return images, labels
    permutation = torch.randperm(len(images), generator=generator).to(images.device)
    axis = int(torch.randint(0, 2, (), generator=generator))
    seam = int(torch.randint(64, 161, (), generator=generator))
    mixed_images, mixed_labels = images.clone(), labels.clone()
    if axis == 0:
        mixed_images[:, :, 112 + seam:, :] = images[permutation, :, 112 + seam:, :]
        mixed_labels[:, seam:, :] = labels[permutation, seam:, :]
    else:
        mixed_images[:, :, :, 112 + seam:] = images[permutation, :, :, 112 + seam:]
        mixed_labels[:, :, seam:] = labels[permutation, :, seam:]
    return mixed_images, mixed_labels


def context_dropout(images, labels, generator):
    assert images.shape[-2:] == (448, 448) and labels.shape[-2:] == (224, 224)
    selected = (torch.rand(len(images), generator=generator) < .5).to(images.device)
    filled = images.mean(dim=(-2, -1), keepdim=True).expand_as(images).clone()
    filled[..., 112:336, 112:336] = images[..., 112:336, 112:336]
    return torch.where(selected[:, None, None, None], filled, images), labels


def illumination_augment(images, labels, generator):
    selected = (torch.rand(len(images), generator=generator) < .5).to(images.device)
    slopes = (torch.rand((len(images), 2), generator=generator) * .2 - .1).to(images.device)
    yy = torch.linspace(-1, 1, images.shape[-2], device=images.device, dtype=images.dtype)[None, None, :, None]
    xx = torch.linspace(-1, 1, images.shape[-1], device=images.device, dtype=images.dtype)[None, None, None, :]
    gain = 1 + slopes[:, 0, None, None, None] * xx + slopes[:, 1, None, None, None] * yy
    shaded = (images * gain).clamp(0, 1)
    return torch.where(selected[:, None, None, None], shaded, images), labels


def gaussian_noise_augment(images, labels, generator):
    augmented = images.clone()
    for index in range(len(images)):
        if float(torch.rand((), generator=generator)) < .5:
            noise = torch.randn((1, images.shape[-2], images.shape[-1]), generator=generator)
            augmented[index] = (images[index] + .02 * noise.to(images.device)).clamp(0, 1)
    return augmented, labels


def intensity_inversion_augment(images, labels, generator):
    augmented = images.clone()
    for index in range(len(images)):
        if float(torch.rand((), generator=generator)) < .5:
            augmented[index] = 1 - images[index]
    return augmented, labels


def histogram_equalization_augment(images, labels, generator):
    augmented = images.clone()
    for index in range(len(images)):
        if float(torch.rand((), generator=generator)) < .5:
            quantized = (images[index, 0].detach().cpu().clamp(0, 1) * 255).round().long()
            counts = torch.bincount(quantized.flatten(), minlength=256)
            cdf = counts.cumsum(0)
            minimum = cdf[counts > 0][0]
            if int(cdf[-1]) > int(minimum):
                lookup = ((cdf - minimum).clamp_min(0).float() / (cdf[-1] - minimum))
                augmented[index] = lookup[quantized].to(images.device).unsqueeze(0).expand_as(images[index])
    return augmented, labels


def shear_augment(images, labels, generator):
    augmented = images.clone()
    targets = labels.clone()
    for index in range(len(images)):
        if float(torch.rand((), generator=generator)) < .5:
            axis = int(torch.randint(2, (), generator=generator))
            coefficient = float(torch.rand((), generator=generator)) * .3 - .15
            matrix = images.new_tensor([[1., 0., 0.], [0., 1., 0.]])
            matrix[axis, 1-axis] = coefficient
            grid = torch.nn.functional.affine_grid(matrix[None], images[index:index+1].shape, align_corners=False)
            augmented[index:index+1] = torch.nn.functional.grid_sample(images[index:index+1], grid, mode="bilinear", padding_mode="reflection", align_corners=False)
            targets[index:index+1] = torch.nn.functional.grid_sample(labels[index:index+1, None].float(), grid, mode="nearest", padding_mode="reflection", align_corners=False)[:, 0].long()
    return augmented, targets


def interior_cutout(images, labels, generator):
    augmented = images.clone()
    for index in range(len(images)):
        if float(torch.rand((), generator=generator)) < .5:
            y = int(torch.randint(images.shape[-2] - 63, (), generator=generator))
            x = int(torch.randint(images.shape[-1] - 63, (), generator=generator))
            augmented[index, :, y:y+64, x:x+64] = images[index].mean()
    return augmented, labels


def mixup_augment(images, labels, generator):
    if torch.rand((), generator=generator) >= 0.5:
        return images, labels
    permutation = torch.randperm(len(images), generator=generator).to(images.device)
    return 0.5 * images + 0.5 * images[permutation], (labels, labels[permutation], 0.5)


def centralize_gradients(model):
    for parameter in model.parameters():
        if parameter.grad is not None and parameter.grad.ndim > 1:
            parameter.grad.sub_(parameter.grad.mean(dim=tuple(range(1, parameter.grad.ndim)), keepdim=True))


def focus_blur_augment(images, labels, generator):
    selected = (torch.rand(len(images), generator=generator) < .5).to(images.device)
    sigma = (torch.rand(len(images), generator=generator) * .9 + .3).to(images.device)
    offsets = torch.arange(-4, 5, device=images.device, dtype=images.dtype)
    kernel = torch.exp(-offsets[None, :] ** 2 / (2 * sigma[:, None] ** 2))
    kernel = kernel / kernel.sum(1, keepdim=True)
    channels = images.shape[1]
    kernel = kernel.repeat_interleave(channels, 0)
    packed = images.reshape(1, len(images) * channels, *images.shape[-2:])
    blurred = torch.nn.functional.conv2d(torch.nn.functional.pad(packed, (4, 4, 0, 0), mode="reflect"), kernel[:, None, None, :], groups=len(images) * channels)
    blurred = torch.nn.functional.conv2d(torch.nn.functional.pad(blurred, (0, 0, 4, 4), mode="reflect"), kernel[:, None, :, None], groups=len(images) * channels).reshape_as(images)
    return torch.where(selected[:, None, None, None], blurred, images), labels


def local_contrast(image):
    mean = torch.nn.functional.avg_pool2d(torch.nn.functional.pad(image, (32, 32, 32, 32), mode="reflect"), 65, stride=1)
    return (image - mean + image.mean(dim=(-2, -1), keepdim=True)).clamp(0, 1)


def use_swin_encoder(model, weights_path, branch):
    from torchvision.models import swin_t
    assert hashlib.sha256(weights_path.read_bytes()).hexdigest() == "704ceda373461b0a224fcdddd75cd2a5e9f8064512ed47adbddef7f343fd147b"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(73035)
        backbone = swin_t(weights=None)
        backbone.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True), strict=True)
        model.encoder = backbone.features
        model.encoder_norm = backbone.norm
        model.swin_encoder = True
        if branch == "R073":
            for name in ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4"):
                delattr(model, name)
            for name, channels in (("decoder4", 1152), ("decoder3", 448), ("decoder2", 224), ("decoder1", 160)):
                block = getattr(model, name)
                block.block[0] = torch.nn.Conv2d(channels, block.block[0].out_channels, 3, padding=1, bias=False)


def make_model(_method="context448", pretrained=True, branch="R052", model_options=None):
    root = Path(__file__).resolve().parents[1]
    assert pretrained
    options = model_options or {}
    if branch == "R052":
        from mimu.convnext_context import ConvNeXtContextUNet
        from torchvision.ops import StochasticDepth
        assert set(options) <= {"stochastic_depth_scale", "decoder_bn_momentum", "multiscale_context", "regional_contrast", "deep_supervision", "freeze_early_encoder", "skip_gate", "nearest_decoder", "decoder_instance_norm", "decoder_silu", "global_context", "spatial_context", "decoder_residual", "scse_max", "local_contrast", "swin_encoder", "full_context_supervision", "encoder_output_stride16", "encoder_silu"}
        model = ConvNeXtContextUNet(
            root / "Thirdparty/weights/convnext_tiny-983f1562.pth",
            "983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d",
        )
        model.full_context_supervision = options.get("full_context_supervision", False)
        if options.get("decoder_residual", False):
            with torch.random.fork_rng(devices=[]):
                for name, block in model.named_children():
                    if name.startswith("decoder"):
                        conv = block.block[0]
                        block.shortcut_projection = torch.nn.Conv2d(conv.in_channels, conv.out_channels, 1, bias=False)
                        torch.nn.init.zeros_(block.shortcut_projection.weight)
        if options.get("local_contrast", False):
            model.input_transform = local_contrast
        if options.get("scse_max", False):
            for name, block in model.named_children():
                if name.startswith("decoder"):
                    block.attention.max_fusion = True
        if options.get("freeze_early_encoder", False):
            model.encoder[:4].requires_grad_(False)
        if options.get("nearest_decoder", False):
            model.decoder_resize_mode = "nearest-exact"
        if options.get("global_context", False):
            with torch.random.fork_rng(devices=[]):
                model.global_context = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),
                                                          torch.nn.Conv2d(256, 256, 1))
                torch.nn.init.zeros_(model.global_context[1].weight)
                torch.nn.init.zeros_(model.global_context[1].bias)
        if options.get("spatial_context", False):
            assert options.get("global_context")
            model.global_context[0] = torch.nn.AdaptiveAvgPool2d((3, 3))
        if options.get("decoder_silu", False):
            for name, layer in list(model.named_modules()):
                if name.startswith("decoder") and ".block." in name and isinstance(layer, torch.nn.ReLU):
                    parent, attribute = name.rsplit('.', 1)
                    setattr(model.get_submodule(parent), attribute, torch.nn.SiLU(inplace=True))
        if options.get("skip_gate", False):
            with torch.random.fork_rng(devices=[]):
                model.skip_gate = torch.nn.Conv2d(128 + 96, 1, 1)
                torch.nn.init.zeros_(model.skip_gate.weight)
                torch.nn.init.zeros_(model.skip_gate.bias)
        for layer in model.modules():
            if isinstance(layer, StochasticDepth):
                layer.p *= options.get("stochastic_depth_scale", 1.0)
        if "decoder_bn_momentum" in options:
            for name, layer in model.named_modules():
                if isinstance(layer, torch.nn.BatchNorm2d):
                    assert name.startswith("decoder")
                    layer.momentum = options["decoder_bn_momentum"]
        if options.get("decoder_instance_norm", False):
            for name, layer in list(model.named_modules()):
                if isinstance(layer, torch.nn.BatchNorm2d):
                    assert name.startswith("decoder")
                    norm = torch.nn.InstanceNorm2d(layer.num_features, eps=layer.eps,
                                                   momentum=layer.momentum, affine=True,
                                                   track_running_stats=True)
                    norm.load_state_dict(layer.state_dict(), strict=True)
                    parent, attribute = name.rsplit('.', 1)
                    setattr(model.get_submodule(parent), attribute, norm)
        if options.get("multiscale_context", False):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(52013)
                model.decoder2.scale_context = MultiScaleContext(64)
        if options.get("regional_contrast", False):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(52015)
                model.projection = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.ReLU(),
                                                       torch.nn.Linear(64, 32))
        if options.get("deep_supervision", False):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(52016)
                model.deep_supervision_head = torch.nn.Conv2d(64, 4, 1)
        if options.get("swin_encoder"):
            use_swin_encoder(model, root / "run/swin_t-704ceda3.pth", branch)
        if options.get("encoder_silu"):
            for name, layer in list(model.encoder.named_modules()):
                if isinstance(layer, torch.nn.GELU):
                    parent, attribute = name.rsplit('.', 1)
                    setattr(model.encoder.get_submodule(parent), attribute, torch.nn.SiLU())
        if options.get("encoder_output_stride16"):
            model.encoder[6][1].stride = (1, 1)
            model.encoder[6][1].dilation = (2, 2)
            model.encoder[6][1].padding = (1, 1)
            for block in model.encoder[7]:
                block.block[0].dilation = (2, 2)
                block.block[0].padding = (6, 6)
        return model
    from mimu.imagenet_context import ImageNetContextUNet
    from mimu.gated_shape import SemanticGatedDetailUNet
    assert branch == "R073"
    assert set(options) <= {"detail_stride", "gate_bias_init", "boundary_auxiliary", "regional_contrast", "feature_consistency", "phase_pair_supervision", "gradient_detail", "local_gate_context", "feature_dropout", "antialias_detail", "geometric_consistency", "joint_detail_gate", "dilated_detail", "detail_silu", "semantic_decoder_silu", "semantic_global_context", "decoder_residual", "scse_max", "local_contrast", "swin_encoder", "full_context_supervision", "encoder_output_stride16", "encoder_silu"}
    model = SemanticGatedDetailUNet(ImageNetContextUNet(
        root / "Thirdparty/weights/resnet101-cd907fc2.pth",
        "cd907fc2a0de2292b2ba2b27d2216bc672d84632d1cf20058d01c0c03b5e020e",
    ), detail_stride=options.get("detail_stride", 2))
    model.full_context_supervision = options.get("full_context_supervision", False)
    model.semantic.full_context_supervision = model.full_context_supervision
    if options.get("swin_encoder"):
        use_swin_encoder(model.semantic, root / "run/swin_t-704ceda3.pth", branch)
    if options.get("decoder_residual", False):
        with torch.random.fork_rng(devices=[]):
            for name, block in model.semantic.named_children():
                if name.startswith("decoder"):
                    conv = block.block[0]
                    block.shortcut_projection = torch.nn.Conv2d(conv.in_channels, conv.out_channels, 1, bias=False)
                    torch.nn.init.zeros_(block.shortcut_projection.weight)
    if options.get("local_contrast", False):
        model.input_transform = local_contrast
    if options.get("scse_max", False):
        for name, block in model.semantic.named_children():
            if name.startswith("decoder"):
                block.attention.max_fusion = True
    if options.get("semantic_global_context", False):
        with torch.random.fork_rng(devices=[]):
            model.semantic.global_context = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Conv2d(256, 256, 1))
            torch.nn.init.zeros_(model.semantic.global_context[1].weight)
            torch.nn.init.zeros_(model.semantic.global_context[1].bias)
    if options.get("semantic_decoder_silu", False):
        for name, layer in list(model.semantic.named_modules()):
            if name.startswith("decoder") and ".block." in name and isinstance(layer, torch.nn.ReLU):
                parent, attribute = name.rsplit('.', 1)
                setattr(model.semantic.get_submodule(parent), attribute, torch.nn.SiLU(inplace=True))
    if options.get("dilated_detail", False):
        model.detail_refiner.dw5.dilation = (2, 2)
        model.detail_refiner.dw5.padding = (4, 4)
    if options.get("detail_silu", False):
        model.detail_refiner.activation = torch.nn.SiLU()
    if options.get("feature_dropout", False):
        model.feature_dropout = torch.nn.Dropout2d(p=.2)
    if options.get("joint_detail_gate", False):
        with torch.random.fork_rng(devices=[]):
            model.semantic_gate.detail_projection = torch.nn.Conv2d(16, 16, 1)
            torch.nn.init.zeros_(model.semantic_gate.detail_projection.weight)
            torch.nn.init.zeros_(model.semantic_gate.detail_projection.bias)
    if options.get("antialias_detail", False):
        taps = torch.tensor([1., 2., 1.])
        model.register_buffer("detail_blur_kernel", (taps[:, None] * taps[None, :])[None, None] / 16)
    if options.get("local_gate_context", False):
        with torch.random.fork_rng(devices=[]):
            model.semantic_gate.local_context = torch.nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False)
            torch.nn.init.zeros_(model.semantic_gate.local_context.weight)
    if options.get("gradient_detail", False):
        with torch.random.fork_rng(devices=[]):
            model.gradient_stem = torch.nn.Conv2d(2, 16, 3, stride=model.detail_stride, padding=1, bias=False)
            torch.nn.init.zeros_(model.gradient_stem.weight)
        model.register_buffer("gradient_kernel", torch.tensor(
            [[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
             [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]]
        ).unsqueeze(1) / 8)
    if "gate_bias_init" in options:
        with torch.no_grad():
            model.semantic_gate.projection.bias.fill_(options["gate_bias_init"])
    if options.get("boundary_auxiliary", False):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(73013)
            model.boundary_head = torch.nn.Sequential(
                torch.nn.Conv2d(32, 16, 3, padding=1), torch.nn.ReLU(),
                torch.nn.Conv2d(16, 4, 1),
            )
    if options.get("regional_contrast", False):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(73015)
            model.projection = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.ReLU(),
                                                   torch.nn.Linear(64, 32))
    if options.get("encoder_silu"):
        model.semantic.relu = torch.nn.SiLU(inplace=True)
        for stage in (model.semantic.layer1, model.semantic.layer2, model.semantic.layer3, model.semantic.layer4):
            for block in stage:
                block.relu = torch.nn.SiLU(inplace=True)

    if options.get("encoder_output_stride16"):
        model.semantic.layer4[0].conv2.stride = (1, 1)
        model.semantic.layer4[0].downsample[0].stride = (1, 1)
        for block in model.semantic.layer4:
            block.conv2.dilation = (2, 2)
            block.conv2.padding = (2, 2)
    model.feature_consistency = options.get("feature_consistency", False)
    model.phase_pair_supervision = options.get("phase_pair_supervision", False)
    model.geometric_consistency = options.get("geometric_consistency", False)
    return model
