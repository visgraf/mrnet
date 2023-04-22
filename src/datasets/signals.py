import warnings
import torch
import scipy.ndimage
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from pathlib import Path

from .constants import SAMPLING_DICT, Sampling
from datasets.sampler import SamplerFactory


class Signal1D(Dataset):
    def __init__(self, data, shape,
                 domain=[-1, 1],
                 attributes=[], 
                 sampling_scheme=Sampling.REGULAR,
                 batch_size=0) -> None:
        
        self.data = data
        self.shape = shape
        self.domain = domain

        self.attributes = attributes
        self.sampler = SamplerFactory.init(sampling_scheme,
                                           self.data,
                                           self.shape,
                                           self.domain,
                                           self.attributes,
                                           batch_size)
        
    def init_fromfile(filepath, domain=[-1, 1], attributes=[],
                      sampling_scheme=Sampling.REGULAR,
                      batch_size=-1):
        data = np.load(filepath)
        shape = [len(data)]

        return Signal1D(data, shape, domain, 
                        attributes, sampling_scheme, batch_size)
    
    def size(self):
        return self.shape
    
    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx):
        item = self.sampler.get_samples(idx)
        return  item
    

        

class ImageSignal(Dataset):
    def __init__(self, data, 
                        width,
                        height,
                        channels=1,
                        sampling_scheme=Sampling.REGULAR,
                        domain=[-1, 1],
                        batch_size=0,
                        attributes=[],
                        domain_mask=None):
        
        # self.image_t = data
        # self.data = torch.flatten(data)
        self.data = data
        self.domain_mask = torch.flatten(domain_mask).bool() if domain_mask else None
        self.attributes = attributes

        self._width = width
        self._height = height
        self.channels = channels
        
        self.domain = domain
        self.sampling_scheme=sampling_scheme
        self.sampler = SamplerFactory.init(sampling_scheme, 
                                           data, 
                                           (width, height),
                                           domain,  attributes,
                                           batch_size)
        self.sampler.make_samples(domain_mask=self.domain_mask)


    def init_fromfile(imagepath, 
                      domain=[-1, 1],
                      channels=3,
                      sampling_scheme='regular',
                      width=None, height=None,
                      attributes=[], 
                      maskpath=None,
                      batch_size=0):
        img = Image.open(imagepath)
        if channels == 1:
            img = img.convert('L')

        if width is not None or height is not None:
            if height is None:
                height = img.height
            if width is None:
                width = img.width
            if width > img.width or height > img.height:
                warnings.warn(f"Resizing to a higher resolution ({width}x{height})", RuntimeWarning)
            img = img.resize((width, height))
        img_tensor = to_tensor(img)
        
        mask = to_tensor(Image.open(maskpath).resize((width, height))) if maskpath else None

        return ImageSignal(img_tensor,
                            img.width,
                            img.height,
                            domain=domain,
                            sampling_scheme=SAMPLING_DICT[sampling_scheme],
                            attributes=attributes,
                            batch_size=batch_size,
                            domain_mask=mask)
    

    def dimensions(self):
        return self._width, self._height

    def image_pil(self):
        return to_pil_image(self.data)

    def image_tensor(self):
        return self.data

    def __sub__(self,other):
        if self.domain != other.domain:
            raise NotImplementedError("Can only operate signals in same domain for now")
        data_self = self.image_t
        data_other = other.image_t
        subtract_data = data_self - data_other
        width,height = self.dimensions()
        return ImageSignal(subtract_data,
                            width,
                            height,
                            domain=self.domain,
                            sampling_scheme=self.sampling_scheme,
                            batch_samples_perc=self.batch_samples_perc,
                            attributes=self.attributes)
    @property
    def batch_size(self):
        return self.sampler.batch_size
                    
    def __len__(self):
        return len(self.sampler)


    def __getitem__(self, idx):
        return self.sampler[idx]