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
                        batch_pixels_perc=None):
        
        self.image_t = data
        self.data = torch.flatten(data)

        self.batch_pixels_perc = batch_pixels_perc
        self._width = width
        self._height = height
        self.channels = channels

        self.sampling_scheme=sampling_scheme
        self.sampler = samplerFactory(sampling_scheme,data)
        self.sampler.make_samples(self.image_t,width,height)


    def init_fromfile(imagepath, sampling_scheme='regular', batch_pixels_perc=None, width=None, height=None):
        img = Image.open(imagepath).convert('L')

        if width is not None or height is not None:
            if height is None:
                height = img.height
            if width is None:
                width = img.width
            img = img.resize((width,height))
        img_tensor = to_tensor(img)

        return ImageSignal(img_tensor,
                            img.width,
                            img.height,
                            sampling_scheme=SAMPLING_DICT[sampling_scheme],
                            batch_pixels_perc=batch_pixels_perc)
    

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
                            batch_pixels_perc=self.batch_pixels_perc)
                    
    def __len__(self):
        return int(1 / self.batch_pixels_perc)

    def __getitem__(self, idx):
        if self.batch_pixels_perc == 1:

            in_dict = {'coords': self.sampler.coords}
            gt_dict = {'d0': self.sampler.img_data.view(-1,1),
                        'd1': self.sampler.img_grad.view(-1,1)
                    }
            return  (in_dict, gt_dict)

        else:
            im_size = self._width*self._height
            rand_idcs = torch.randint(im_size, size=(1, int(self.batch_pixels_perc*im_size)))
            rand_coords = self.sampler.coords[rand_idcs, :]
            
            d0 = self.sampler.img_data.view(-1,1)
            rand_d0 = d0[rand_idcs, :]
            
            d1 = self.sampler.img_grad.view(-1,1)
            rand_d1 = d1[rand_idcs, :]

            in_dict = {'idx':idx,'coords':rand_coords}
            gt_dict = {'d0': rand_d0, 'd1': rand_d1}

            return  (in_dict,gt_dict)

# OBS: in the future consider to replace the stored self.data with tensor format self.data.view(-1,1)
#      (the same for all attributes, i.e. d1, etc...)
