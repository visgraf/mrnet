import warnings
import torch
import scipy.ndimage
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from pathlib import Path

from .constants import SAMPLING_DICT, Sampling
from datasets.sampler import samplerFactory



def make_mask(srcpath, mask_color):
    img = np.array(Image.open(srcpath))
    mask = img != mask_color
    path = Path(srcpath)
    path = path.parent.absolute().joinpath("mask.png")
    Image.fromarray(mask).save(path)
    return str(path)

class ImageSignal(Dataset):
    def __init__(self, data, 
                        width,
                        height,
                        channels=1,
                        sampling_scheme=Sampling.REGULAR,
                        domain=[-1, 1],
                        batch_samples_perc=None,
                        attributes=[],
                        domain_mask=None):
        
        self.image_t = data
        self.data = torch.flatten(data)
        self.domain_mask = torch.flatten(domain_mask).bool() if domain_mask else None
        self.attributes = attributes

        self.batch_samples_perc = batch_samples_perc
        self._width = width
        self._height = height
        self.channels = channels
        
        self.domain = domain
        self.sampling_scheme=sampling_scheme
        self.sampler = samplerFactory(sampling_scheme, data, attributes)
        self.sampler.make_samples(self.image_t, width, height, 
                                  domain, self.batch_samples_perc, 
                                  domain_mask=self.domain_mask)


    def init_fromfile(imagepath, 
                      domain=[-1, 1],
                      batch_samples_perc=None, 
                      sampling_scheme='regular',
                      width=None, height=None,
                      attributes=[], channels=3,
                      maskpath=None):
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
                            batch_samples_perc=batch_samples_perc,
                            attributes=attributes,
                            domain_mask=mask)
    

    def dimensions(self):
        return self._width, self._height

    def image_pil(self):
        return to_pil_image(self.image_t)

    def image_tensor(self):
        return self.image_t

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
                    
    def __len__(self):
        return self.sampler.total_size()


    def __getitem__(self, idx):
        item = self.sampler.get_samples(idx)
        return  item

