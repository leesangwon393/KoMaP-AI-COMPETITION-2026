"""Build the released R052/R073 recipes and their train-only helpers.

Historical candidate experiments are documented in docs/experiments/.
Model-specific modules retain the released state-dict layout and RNG order.
"""
from pathlib import Path

import torch


def interior_cutout(images, labels, generator):
    augmented = images.clone()
    for index in range(len(images)):
        if float(torch.rand((), generator=generator)) < .5:
            y = int(torch.randint(images.shape[-2] - 63, (), generator=generator))
            x = int(torch.randint(images.shape[-1] - 63, (), generator=generator))
            augmented[index, :, y:y+64, x:x+64] = images[index].mean()
    return augmented, labels

def centralize_gradients(model):
    for parameter in model.parameters():
        if parameter.grad is not None and parameter.grad.ndim > 1:
            parameter.grad.sub_(parameter.grad.mean(dim=tuple(range(1, parameter.grad.ndim)), keepdim=True))

def make_model(_method="context448", pretrained=True, branch="R052", model_options=None):
    """Construct a final model using separately supplied ImageNet weights."""
    if not pretrained:
        raise ValueError("Model 1/2 require the released ImageNet initialization weights")
    root = Path(__file__).resolve().parents[1]
    options = model_options or {}
    if branch == "R052":
        from mimu.convnext_context import ConvNeXtContextUNet

        allowed = {"decoder_bn_momentum", "decoder_silu", "global_context",
                   "decoder_residual", "full_context_supervision"}
        if set(options) - allowed:
            raise ValueError(f"Unsupported final model1 options: {set(options) - allowed}")
        model = ConvNeXtContextUNet(
            root / "Thirdparty/weights/convnext_tiny-983f1562.pth",
            "983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d",
        )
        model.full_context_supervision = options.get("full_context_supervision", False)
        # Preserve the released order of module construction and initialization.
        if options.get("decoder_residual", False):
            with torch.random.fork_rng(devices=[]):
                for name, block in model.named_children():
                    if name.startswith("decoder"):
                        conv = block.block[0]
                        block.shortcut_projection = torch.nn.Conv2d(
                            conv.in_channels, conv.out_channels, 1, bias=False
                        )
                        torch.nn.init.zeros_(block.shortcut_projection.weight)
        if options.get("global_context", False):
            with torch.random.fork_rng(devices=[]):
                model.global_context = torch.nn.Sequential(
                    torch.nn.AdaptiveAvgPool2d(1), torch.nn.Conv2d(256, 256, 1)
                )
                torch.nn.init.zeros_(model.global_context[1].weight)
                torch.nn.init.zeros_(model.global_context[1].bias)
        if options.get("decoder_silu", False):
            for name, layer in list(model.named_modules()):
                if name.startswith("decoder") and ".block." in name and isinstance(layer, torch.nn.ReLU):
                    parent, attribute = name.rsplit('.', 1)
                    setattr(model.get_submodule(parent), attribute, torch.nn.SiLU(inplace=True))
        if "decoder_bn_momentum" in options:
            for name, layer in model.named_modules():
                if isinstance(layer, torch.nn.BatchNorm2d):
                    if not name.startswith("decoder"):
                        raise ValueError(f"Unexpected BatchNorm outside decoder: {name}")
                    layer.momentum = options["decoder_bn_momentum"]
        return model

    if branch == "R073":
        from mimu.imagenet_context import ImageNetContextUNet
        from mimu.gated_shape import SemanticGatedDetailUNet

        if set(options) - {"full_context_supervision"}:
            raise ValueError(f"Unsupported final model2 options: {set(options)}")
        model = SemanticGatedDetailUNet(ImageNetContextUNet(
            root / "Thirdparty/weights/resnet101-cd907fc2.pth",
            "cd907fc2a0de2292b2ba2b27d2216bc672d84632d1cf20058d01c0c03b5e020e",
        ))
        model.full_context_supervision = options.get("full_context_supervision", False)
        model.semantic.full_context_supervision = model.full_context_supervision
        model.feature_consistency = False
        model.phase_pair_supervision = False
        model.geometric_consistency = False
        return model

    raise ValueError(f"Unknown final model branch: {branch}")
