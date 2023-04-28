import warnings
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image

from .constants import SAMPLING_DICT, Sampling
from datasets.sampler import SamplerFactory
    

class BaseSignal(Dataset):
    def __init__(self, data,
                 domain=[-1, 1],
                 attributes=[], 
                 sampling_scheme=Sampling.REGULAR,
                 batch_size=0):
        
        self.data = data
        self.domain = domain
        self.attributes = attributes
        if isinstance(sampling_scheme, str):
            sampling_scheme = SAMPLING_DICT[sampling_scheme]
        self.sampler = SamplerFactory.init(sampling_scheme,
                                           self.data,
                                           self.domain,
                                           self.attributes,
                                           batch_size)
    def size(self):
        return self.data.shape
    
    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx):
        item = self.sampler[idx]
        return  item
    
    @property
    def batch_size(self):
        return self.sampler.batch_size
    
    def type_code(self):
        return 'B'
    
    def new_like(other, data, attributes=[]):
        if other.type_code() == '1D':
            Class = Signal1D
        elif other.type_code() == '2D':
            Class = ImageSignal
        else:
            Class = BaseSignal
        return Class(data, other.domain,
                    attributes, other.sampler.scheme(),
                    other.sampler.batch_size)
        

class Signal1D(BaseSignal):
    def init_fromfile(filepath, 
                      domain=[-1, 1], 
                      attributes=[],
                      sampling_scheme=Sampling.REGULAR,
                      batch_size=-1):
        
        data = np.load(filepath)
        if isinstance(sampling_scheme, str):
            sampling_scheme = SAMPLING_DICT[sampling_scheme]

        return Signal1D(data.view(1, -1), domain, 
                        attributes, sampling_scheme, batch_size)
    
    def type_code(self):
        return "1D"


class ImageSignal(BaseSignal):
    def init_fromfile(imagepath, 
                      domain=[-1, 1],
                      channels=3,
                      sampling_scheme=Sampling.REGULAR,
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
        if isinstance(sampling_scheme, str):
            sampling_scheme = SAMPLING_DICT[sampling_scheme]

        return ImageSignal(img_tensor,
                            domain=domain,
                            sampling_scheme=sampling_scheme,
                            attributes=attributes,
                            batch_size=batch_size)
                            #domain_mask=mask)

    def image_pil(self):
        return to_pil_image(self.data)
    
    def type_code(self):
        return "2D"

    def __sub__(self,other):
        if self.domain != other.domain:
            raise NotImplementedError("Can only operate signals in same domain for now")
        data_self = self.image_t
        data_other = other.image_t
        subtract_data = data_self - data_other
        width,height = self.dimensions()
        return ImageSignal(subtract_data,
                            domain=self.domain,
                            sampling_scheme=self.sampling_scheme,
                            batch_samples_perc=self.batch_samples_perc,
                            attributes=self.attributes)