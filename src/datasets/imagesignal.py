import torch
import scipy.ndimage
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image

from .constants import SAMPLING_DICT,Sampling
from datasets.sampler import samplerFactory

class ImageSignal(Dataset):
    def __init__(self, data, 
                        width,
                        height,
                        channels=1,
                        sampling_scheme=Sampling.REGULAR,
                        batch_samples_perc=None,
                        attributes=[]):
        
        self.image_t = data
        self.data = torch.flatten(data)
        self.attributes = attributes

        self.batch_samples_perc = batch_samples_perc
        self._width = width
        self._height = height
        self.channels = channels

        self.sampling_scheme=sampling_scheme
        self.sampler = samplerFactory(sampling_scheme, data, attributes)
        self.sampler.make_samples(self.image_t,width,height, self.batch_samples_perc)


    def init_fromfile(imagepath, batch_samples_perc=None, sampling_scheme='regular', width=None, height=None, attributes=[]):
        img = Image.open(imagepath).convert('L')

        if width is not None or height is not None:
            if height is None:
                height = img.height
            if width is None:
                width = img.width
            img = img.resize((width, height))
        img_tensor = to_tensor(img)

        return ImageSignal(img_tensor,
                            img.width,
                            img.height,
                            sampling_scheme=SAMPLING_DICT[sampling_scheme],
                            batch_samples_perc=batch_samples_perc,
                            attributes=attributes)
    

    def dimensions(self):
        return self._width, self._height

    def image_pil(self):
        return to_pil_image(self.image_t)

    def image_tensor(self):
        return self.image_t

    def __sub__(self,other):
        data_self = self.image_t
        data_other = other.image_t
        subtract_data = data_self - data_other
        width,height = self.dimensions()
        return ImageSignal(subtract_data,
                            width,
                            height,
                            sampling_scheme=self.sampling_scheme,
                            batch_samples_perc=self.batch_samples_perc,
                            attributes=self.attributes)
                    
    def __len__(self):
        return int(1.0 / self.batch_samples_perc)

    def __getitem__(self, idx):
        item = self.sampler.get_samples(idx)
        return  item

