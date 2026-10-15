"""Released full U2_ECA model with central supervision."""
import torch
from torch import nn
from torch.nn import functional as F
from .rsu_models import FullU2Net


def resize(x, reference):
    size = reference.shape[-2:] if isinstance(reference, torch.Tensor) else reference
    return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

def center_half(x):
    h, w = x.shape[-2:]
    if h % 4 or w % 4:
        raise ValueError('Center feature crop requires dimensions divisible by four')
    return x[..., h//4:3*h//4, w//4:3*w//4]


class SegmentationModel(nn.Module):
    def __init__(self,config,pretrained=False):
        super().__init__();m=config['model'];self.kind=m['architecture'];self.target_size=config['train']['target']
        expected = {'architecture': 'u2net', 'initialization': 'none', 'attention': 'none',
                    'u2_rsu_attention': 'eca', 'u2_decoder_stages': 'none',
                    'u2_fusion_attention': 'none', 'deep_supervision': False,
                    'hybrid_backbone': 'none', 'context': 'none', 'bem': False,
                    'contrast': False, 'connectivity': False, 'rrcu': 'none',
                    'skip_fusion': 'none', 'upsampling': 'bilinear'}
        if pretrained or any(m.get(key) != value for key,value in expected.items()):
            raise ValueError('Only the released, scratch-initialized U2_ECA is supported')
        self.network=FullU2Net(config['train']['seed'])
        self.register_buffer('input_mean',torch.tensor(config['normalization']['mean']).reshape(1,3,1,1))
        self.register_buffer('input_std',torch.tensor(config['normalization']['std']).reshape(1,3,1,1))
    def forward(self,x,return_features=False):
        if self.training and x.shape[-2:]!=(2*self.target_size,2*self.target_size):raise ValueError('Unexpected training context')
        z=(x.repeat(1,3,1,1)-self.input_mean)/self.input_std;aux={};pred=self.network(z)
        pred,_=pred
        pred=resize(pred,x.shape[-2:])
        if self.training:pred=center_half(pred)
        return (pred,aux) if return_features else pred
