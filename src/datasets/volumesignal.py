import warnings
import torch
import scipy.ndimage
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from pathlib import Path

from .constants import SAMPLING_DICT, Sampling
from datasets.sampler3d import RegularSampler3D


class VolumeSignal(Dataset):
    def __init__(self, data, 
                        width,
                        height,
                        depth,
                        channels=1,
                        sampling_scheme=Sampling.REGULAR,
                        domain=[-1, 1],
                        batch_samples_perc=None,
                        attributes=[],
                        domain_mask=None):
        
        self.volume = data
        self.data = torch.flatten(data)
        self.domain_mask = torch.flatten(domain_mask).bool() if domain_mask else None
        self.attributes = attributes

        self.batch_samples_perc = batch_samples_perc
        self._width = width
        self._height = height
        self._depth = depth
        self.channels = channels
        
        self.domain = domain
        self.sampling_scheme=sampling_scheme
        self.sampler = RegularSampler3D(self.volume, self.attributes)
        self.sampler.make_samples(self.volume, width, height, depth,
                                  domain, self.batch_samples_perc, 
                                  domain_mask=self.domain_mask)


    def init_fromfile(volumepath, 
                      domain=[-1, 1],
                      batch_samples_perc=None, 
                      sampling_scheme='regular',
                      width=None, height=None,
                      attributes=[], channels=3,
                      maskpath=None):
        volume = np.load(volumepath).astype(np.float32)
        
        if channels == 1:
            volume = (0.2126 * volume[:, :, :, 0] 
                      + 0.7152 * volume[:, :, :, 1] 
                      + 0.0722 * volume[:, :, :, 2])
            volume = np.expand_dims(volume, axis=0)
        else:
            if volume.shape[-1] == 3:
                volume.transpose((3, 0, 1, 2))

        if width is not None or height is not None:
            raise NotImplementedError("Can't resize volume at this moment")
        vol_tensor = torch.from_numpy(volume)
        w, h, d = volume.shape[1:]
        
        # mask = to_tensor(Image.open(maskpath).resize((width, height))) if maskpath else None
        mask = None

        return VolumeSignal(vol_tensor,
                            w, h, d,
                            domain=domain,
                            sampling_scheme=SAMPLING_DICT[sampling_scheme],
                            batch_samples_perc=batch_samples_perc,
                            attributes=attributes,
                            domain_mask=mask)
    

    def dimensions(self):
        return self._width, self._height, self._depth

    def volume_tensor(self):
        return self.volume

    def __sub__(self, other):
        raise NotImplementedError("Can't operate volumes for now")
                    
    def __len__(self):
        return self.sampler.total_size()


    def __getitem__(self, idx):
        item = self.sampler.get_samples(idx)
        return  item

