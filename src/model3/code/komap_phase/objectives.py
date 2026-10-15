"""Train-label-only auxiliary objectives; no validation/test pseudo labels."""
import torch
from torch import nn
from torch.nn import functional as F

from .loss import R022Loss


def contact_band(target, radius=2):
    """Pixels of classes1/2 around their direct 4-neighbor contact, no wraparound."""
    padded = F.pad(target, (1, 1, 1, 1), value=255)
    contact = torch.zeros_like(target, dtype=torch.bool)
    h, w = target.shape[-2:]
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        neighbor = padded[..., 1+dy:1+dy+h, 1+dx:1+dx+w]
        contact |= ((target == 1) & (neighbor == 2)) | ((target == 2) & (neighbor == 1))
    if radius:
        contact = F.max_pool2d(contact.float()[:, None], 2*radius+1, stride=1, padding=radius)[:, 0] > 0
    return contact & ((target == 1) | (target == 2))


class PhaseLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.base = R022Loss(**config['loss'])
        self.aux = config['aux']
        self.generator = torch.Generator(device='cpu').manual_seed(config['train']['seed']+2103)

    def forward(self, logits, target, embeddings, epoch):
        extra = embeddings if isinstance(embeddings, dict) else {}
        embeddings = [] if isinstance(embeddings, dict) else embeddings
        base_loss = self.base(logits, target)
        contact_fraction = logits.new_zeros(())
        if self.aux['contact_weight'] != 1 or any(self.aux[k] for k in
                ('contrast_weight','connectivity_weight','deep_weight','consistency_weight','ocr_weight')):
            raise ValueError('Auxiliary experiment losses are not part of U2_ECA')
        ramp = max(0., min(1., (epoch-self.aux['warmup_epochs'])/self.aux['ramp_epochs']))
        weight = self.aux['contrast_weight']*ramp
        local = logits.new_zeros(())
        points = 0
        connection=logits.new_zeros(())
        deep=logits.new_zeros(());consistency=logits.new_zeros(());ocr=logits.new_zeros(())
        total = base_loss + weight*local + self.aux['connectivity_weight']*connection
        total=total+self.aux['deep_weight']*deep+self.aux['consistency_weight']*consistency+self.aux['ocr_weight']*ocr
        return total, {'seg_loss': float(base_loss.detach()), 'contrast_loss': float(local.detach()),
                       'contrast_lambda': weight, 'contrast_points': points,
                       'contact_fraction': float(contact_fraction.detach()),'connectivity_loss':float(connection.detach()),
                       'deep_loss':float(deep.detach()),'consistency_loss':float(consistency.detach()),'ocr_loss':float(ocr.detach())}
