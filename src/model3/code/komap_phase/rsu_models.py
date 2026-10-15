"""Full U²-Net with ECA on all outer RSU residual branches.

Based on xuebinqin/U-2-Net (Apache-2.0). Upstream blocks remain unchanged
in vendor_u2net.py; attribution and license are in resources/licenses/.
"""
import torch
from torch import nn
from torch.nn import functional as F

from . import vendor_u2net as upstream


def resize(x,size):
    return F.interpolate(x,size=size,mode='bilinear',align_corners=False)


class ECA(nn.Module):
    def __init__(self,c):
        super().__init__();self.conv=nn.Conv1d(1,1,3,padding=1,bias=False)
    def forward(self,x):
        gate=self.conv(x.mean((-2,-1))[:,None]).sigmoid()[:,0,:,None,None]
        return x*gate


class FullU2Net(upstream.U2NET):
    """Keep upstream feature widths/6-head fusion; return logits, never sigmoid.

    ECA is applied inside every outer RSU before residual addition.
    """
    def __init__(self,seed):
        super().__init__(in_ch=3,out_ch=4)
        # Preserve state-dict layout and the released ECA initialization stream.
        self.attentions=nn.ModuleList([nn.Identity() for _ in range(5)])
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed+4102)
            self.rsu_eca=nn.ModuleList([ECA(c) for c in (64,128,256,512,512,512,512,256,128,64,64)])
        self.fusion_attentions=nn.ModuleList([nn.Identity() for _ in range(6)])
        self.eca_handles=[]
        if self.rsu_eca:
            stages=[getattr(self,f'stage{i}') for i in range(1,7)]+[getattr(self,f'stage{i}d') for i in range(5,0,-1)]
            for index,stage in enumerate(stages):
                # Transform the learned residual branch before adding hxin.
                # The untouched upstream RSU forward computes ECA(hx1d)+hxin.
                def recalibrate(_module,_inputs,output,index=index):
                    return self.rsu_eca[index](output)
                self.eca_handles.append(stage.rebnconv1d.register_forward_hook(recalibrate))

    def forward(self,x):
        encoder=[]
        z=x
        for i in range(1,7):
            z=getattr(self,f'stage{i}')(z);encoder.append(z)
            if i<6: z=getattr(self,f'pool{i}{i+1}')(z)
        decoded=[]
        for index,i in enumerate(range(5,0,-1)):
            skip=encoder[i-1]
            z=getattr(self,f'stage{i}d')(torch.cat([resize(z,skip.shape[-2:]),skip],1))
            z=self.attentions[index](z);decoded.append(z)
        features=list(reversed(decoded))+[encoder[-1]]
        if hasattr(self,'fusion_attentions'):
            features=[gate(feature) for gate,feature in zip(self.fusion_attentions,features)]
        sides=[resize(getattr(self,f'side{i}')(feature),x.shape[-2:])
               for i,feature in enumerate(features,1)]
        fused=self.outconv(torch.cat(sides,1))
        return fused,sides
