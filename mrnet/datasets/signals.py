import warnings
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from mrnet.datasets.sampler import RegularSampler, SAMPLING_CLASSES
from scipy.ndimage import sobel


def compute_derivatives1D(data):
    dims = len(data.shape[1:])
    directions = []
    for d in range(1, dims + 1):
        directions.append(
            torch.from_numpy(sobel(data, d, mode='wrap'))
            )
    # channels x N x dims
    return  torch.stack(directions, dim=-1)

def compute_derivatives2d(data):
        dims = len(data.shape[1:])
        directions = []
        for d in range(1, dims + 1):
            directions.append(
                torch.from_numpy(sobel(data, d, mode='wrap'))
                )
        return torch.stack(directions, dim=-1)

class BaseSignal(Dataset):
    def __init__(self, data,
                 domain=[-1, 1],
                 attributes={}, 
                 SamplerClass=RegularSampler,
                 batch_size=0,
                 shuffle=True,
                 **kwargs):
        
        self.data = data
        self.domain = domain
        self.color_space = kwargs.get('color_space', 'L')

        self.sampler = SamplerClass(self.data,
                                    self.domain,
                                    attributes,
                                    batch_size,
                                    shuffle)

    def size(self):
        return self.data.shape
    
    def add_mask(self, mask):
        self.sampler.add_mask(mask)
    
    @property
    def shape(self):
        return self.size()
    
    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx):
        return self.sampler[idx]
    
    @property
    def batch_size(self):
        return self.sampler.batch_size
    
    @property
    def coords(self):
        return self.sampler.coords
    
    @property
    def attributes(self):
        return self.sampler.attributes
    
    @attributes.setter
    def attributes(self, value):
        self.sampler.attributes = value
    
    @property
    def domain_mask(self):
        return self.sampler.mask
    
    def type_code(self):
        return 'B'
    
    def new_like(other, data, shuffle=None):
        if other.type_code() == '1D':
            Class = Signal1D
        elif other.type_code() == '2D':
            Class = ImageSignal
        else:
            Class = BaseSignal
        if shuffle is None:
            shuffle = other.sampler.shuffle
        return Class(data, other.domain,
                    other.attributes, 
                    other.sampler.__class__,
                    other.sampler.batch_size,
                    shuffle=shuffle,
                    color_space=other.color_space)
        

class Signal1D(BaseSignal):
    def init_fromfile(filepath, 
                      domain=[-1, 1], 
                      attributes={},
                      sampling_scheme="regular",
                      batch_size=-1,
                      **kwargs):
        
        data = np.load(filepath)
        sampler_class = kwargs.get('sampler_class', 
                                   SAMPLING_CLASSES[sampling_scheme])

        return Signal1D(torch.from_numpy(data).float().view(1, -1), 
                        domain, 
                        attributes,
                        sampler_class,
                        batch_size)
    
    def type_code(self):
        return "1D"


class ImageSignal(BaseSignal):

    def init_fromfile(imagepath, 
                      domain=[-1, 1],
                      attributes={},
                      sampling_scheme="regular",
                      batch_size=0,
                      color_space='RGB',
                      **kwargs):
        img = Image.open(imagepath)
        img.mode
        if color_space != img.mode:
            img = img.convert(color_space)

        width = kwargs.get('width', 0)
        height = kwargs.get('height', 0)
        if width or height:
            if not height:
                height = img.height
            if not width:
                width = img.width
            if width > img.width or height > img.height:
                warnings.warn(f"Resizing to a higher resolution ({width}x{height})", RuntimeWarning)
            img = img.resize((width, height))
        img_tensor = to_tensor(img)
        
        sampler_class = kwargs.get('sampler_class', 
                                   SAMPLING_CLASSES[sampling_scheme])

        return ImageSignal(img_tensor,
                            domain=domain,
                            attributes=attributes,
                            SamplerClass=sampler_class,
                            batch_size=batch_size,
                            color_space=color_space)

    
    def load_mask(self, maskpath):
        mask = to_tensor(Image.open(maskpath)).squeeze(0).bool()
        self.add_mask(mask)

    def image_pil(self):
        return to_pil_image(self.data)
    
    def type_code(self):
        return "2D"