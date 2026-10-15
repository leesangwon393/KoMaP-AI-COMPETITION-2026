"""Native-resolution context patch dataset base."""
import numpy as np
import torch
from .data import NativePatchDataset



class ContextDataset(NativePatchDataset):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.padded=[np.pad(im,112,mode='reflect') for im in self.images]

    def __getitem__(self,index):
        # Preserve the exact random draw sequence and target crops of NativePatchDataset.
        if torch.rand(()) < self.enriched_probability:
            sources=tuple(self.enriched)
            source=sources[int(torch.randint(len(sources),()).item())]
            tiles=self.enriched[source]
        else:
            source=int(torch.randint(len(self.paths),()).item())
            tiles=self.origins[source]
        y,x=tiles[int(torch.randint(len(tiles),()).item())]
        image=torch.from_numpy(self.padded[source][y:y+448,x:x+448].copy()).float()[None]/255.
        target=torch.from_numpy(self.labels[source][y:y+224,x:x+224].astype(np.int64,copy=True))
        k=int(torch.randint(4,()).item())
        image,target=image.rot90(k,(-2,-1)),target.rot90(k,(-2,-1))
        if bool(torch.randint(2,()).item()):image,target=image.flip(-1),target.flip(-1)
        return image,target,source
