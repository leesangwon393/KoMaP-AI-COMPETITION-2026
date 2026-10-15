"""Per-image Lovasz-Softmax IoU surrogate alongside the ZIP CE/Dice loss."""
import torch
from torch import nn
from torch.nn import functional as F

from .loss import HybridLoss


def lovasz_gradient(labels_sorted):
    positives = labels_sorted.sum()
    intersection = positives - labels_sorted.cumsum(0)
    union = positives + (1 - labels_sorted).cumsum(0)
    jaccard = 1 - intersection / union.clamp_min(1)
    return torch.cat((jaccard[:1], jaccard[1:] - jaccard[:-1]))


class LovaszCompositeLoss(nn.Module):
    """Keep CE=.5; replace half of weighted Dice=.5 with per-image IoU=.25."""

    def __init__(self, pixel_loss="ce", pixel_weight=.5, dice_weight=.5,
                 class_weights=(1, 1, 2, 1), lovasz_all_classes=False, squared_dice=False):
        super().__init__()
        if pixel_loss != 'ce':
            raise ValueError('The final recipe uses unweighted CE')
        self.squared_dice = squared_dice
        self.lovasz_all_classes = lovasz_all_classes
        self.base = HybridLoss(pixel_loss=pixel_loss, pixel_weight=pixel_weight,
                               dice_weight=dice_weight / 2, class_weights=class_weights)
        self.lovasz_weight = dice_weight / 2

    def forward(self, logits, target):
        probs = F.softmax(logits.float(), dim=1)
        per_image = []
        for image, labels in zip(probs, target):
            per_class, weights = [], []
            for cls, weight in enumerate(self.base.class_weights):
                foreground = (labels == cls).reshape(-1).float()
                if not bool(foreground.any()) and not self.lovasz_all_classes:
                    continue
                errors = (foreground - image[cls].reshape(-1)).abs()
                sorted_errors, order = errors.sort(descending=True)
                per_class.append(torch.dot(sorted_errors, lovasz_gradient(foreground[order])) * weight)
                weights.append(weight)
            per_image.append(torch.stack(per_class).sum() / torch.stack(weights).sum())
        base = self.base(logits, target)
        if self.squared_dice:
            one_hot = F.one_hot(target, 4).permute(0, 3, 1, 2).to(probs.dtype)
            intersection = (probs * one_hot).sum(dim=(2, 3))
            predicted = probs.sum(dim=(2, 3))
            actual = one_hot.sum(dim=(2, 3))
            dice = (2 * intersection + 1e-5) / (predicted + actual + 1e-5)
            squared = (2 * intersection + 1e-5) / (probs.square().sum(dim=(2, 3)) + one_hot.square().sum(dim=(2, 3)) + 1e-5)
            difference = ((self.base.class_weights * (dice - squared)).sum(-1) / self.base.class_weights.sum()).mean()
            base = base + self.base.dice_weight * difference
        return base + self.lovasz_weight * torch.stack(per_image).mean()
