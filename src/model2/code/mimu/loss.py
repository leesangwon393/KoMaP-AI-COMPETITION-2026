import torch
from torch import nn
from torch.nn import functional as F


class HybridLoss(nn.Module):
    """Unweighted cross-entropy and class-weighted soft Dice for the final recipe."""
    def __init__(self, dice_weight=2.0,
                 class_weights=(1.0, 1.0, 1.0, 1.0),
                 pixel_loss='ce', pixel_weight=1.0):
        super().__init__()
        if pixel_loss != 'ce':
            raise ValueError('The final recipe uses CE')
        self.dice_weight, self.pixel_loss, self.pixel_weight = dice_weight, pixel_loss, pixel_weight
        if len(class_weights) != 4 or any(v <= 0 for v in class_weights):
            raise ValueError('class_weights must contain four positive values')
        weights = torch.tensor(class_weights, dtype=torch.float32)
        self.register_buffer('class_weights', weights / weights.mean())

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1)
        p = logp.exp()
        pixel = F.cross_entropy(logits, target)
        y = F.one_hot(target, logits.shape[1]).permute(0, 3, 1, 2).to(p.dtype)
        dims = (2, 3)
        dice = (2 * (p * y).sum(dims) + 1e-5) / (p.sum(dims) + y.sum(dims) + 1e-5)
        weighted_dice_loss = (((1 - dice) * self.class_weights).sum(-1) / self.class_weights.sum()).mean()
        return self.pixel_weight * pixel + self.dice_weight * weighted_dice_loss
