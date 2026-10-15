"""Reflect-padded 448-input, central-224 probability inference."""
import numpy as np
import torch
from mimu.data import _patch_origins


@torch.inference_mode()
def predict(model,image,device,tta=True):
    h,w=image.shape
    padded=np.pad(image,112,mode='reflect')
    origins=_patch_origins(h,w,224,112)
    sums=torch.zeros(2,4,h,w)
    coverage=torch.zeros(h,w)
    for start in range(0,len(origins),4):
        loc=origins[start:start+4]
        x=torch.stack([torch.from_numpy(padded[y:y+448,z:z+448].copy()).float()[None]/255 for y,z in loc]).to(device)
        combined=torch.zeros(len(loc),4,224,224)
        original=None
        for k in range(4) if tta else range(1):
            for flip in (False,True) if tta else (False,):
                view=x.rot90(k,(-2,-1))
                if flip:view=view.flip(-1)
                prob=model(view).softmax(1)
                if flip:prob=prob.flip(-1)
                prob=prob.rot90(-k,(-2,-1))[...,112:336,112:336].cpu()
                combined+=prob/(8 if tta else 1)
                if k==0 and not flip:original=prob
        for j,(y,z) in enumerate(loc):
            sums[0,:,y:y+224,z:z+224]+=original[j]
            sums[1,:,y:y+224,z:z+224]+=combined[j]
            coverage[y:y+224,z:z+224]+=1
    return sums/coverage.clamp_min(1)[None,None]
