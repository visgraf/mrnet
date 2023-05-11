import warnings
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from pathlib import Path
from .constants import SAMPLING_DICT, Sampling
from datasets.sampler import SamplerFactory
from scipy.ndimage import sobel


def make_mask(srcpath, mask_color):
    img = np.array(Image.open(srcpath))
    mask = img != mask_color
    path = Path(srcpath)
    path = path.parent.absolute().joinpath("mask.png")
    Image.fromarray(mask).save(path)
    return str(path)

class BaseSignal(Dataset):
    def __init__(self, data,
                 domain=[-1, 1],
                 attributes=[], 
                 sampling_scheme=Sampling.REGULAR,
                 batch_size=0,
                 shuffle=True):
        
        self.data = data
        self.domain = domain
        self.attributes = attributes
        if isinstance(sampling_scheme, str):
            sampling_scheme = SAMPLING_DICT[sampling_scheme]
        self.sampler = SamplerFactory.init(sampling_scheme,
                                           self.data,
                                           self.domain,
                                           self.attributes,
                                           batch_size,
                                           shuffle)
    def compute_derivatives(self):
        dims = len(self.data.shape[1:])
        directions = []
        for d in range(1, dims + 1):
            directions.append(
                torch.from_numpy(sobel(self.data, d, mode='wrap'))
                )
        # channels x N x dims
        self.data_attributes = {'d1': torch.stack(directions, dim=-1)}

    def size(self):
        return self.data.shape
    
    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx):
        return self.sampler[idx]
    
    @property
    def batch_size(self):
        return self.sampler.batch_size
    
    def type_code(self):
        return 'B'
    
    def new_like(other, data, attributes=[], shuffle=None):
        if other.type_code() == '1D':
            Class = Signal1D
        elif other.type_code() == '2D':
            Class = ImageSignal
        elif other.type_code() == '3D':
            Class = VolumeSignal
        else:
            Class = BaseSignal
        if shuffle is None:
            shuffle = other.sampler.shuffle
        return Class(data, other.domain,
                    attributes, other.sampler.scheme(),
                    other.sampler.batch_size,
                    shuffle)
        

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
    
class VolumeSignal(BaseSignal):

    def init_fromfile(volumepath, 
                      domain=[-1, 1],
                      sampling_scheme=Sampling.REGULAR,
                      width=None, height=None,
                      attributes=[], channels=3,
                      batch_size=0,
                      maskpath=None):
        volume = np.load(volumepath).astype(np.float32)
        
        if channels == 1:
            volume = (0.2126 * volume[:, :, :, 0] 
                      + 0.7152 * volume[:, :, :, 1] 
                      + 0.0722 * volume[:, :, :, 2])
            volume = np.expand_dims(volume, axis=0)
        else:
            if volume.shape[-1] == 3:
                volume = volume.transpose((3, 0, 1, 2))

        if width is not None or height is not None:
            raise NotImplementedError("Can't resize volume at this moment")
        vol_tensor = torch.from_numpy(volume)
        # mask = to_tensor(Image.open(maskpath).resize((width, height))) if maskpath else None

        if isinstance(sampling_scheme, str):
            sampling_scheme = SAMPLING_DICT[sampling_scheme]
        mask = None
        print(vol_tensor.shape, 'VOLUME SHAPE')
        return VolumeSignal(vol_tensor,
                            domain=domain,
                            sampling_scheme=sampling_scheme,
                            attributes=attributes,
                            batch_size=batch_size)
                            #domain_mask=mask)

    def type_code(self):
        return "3D"