import torch
import scipy.ndimage
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from .constants import Sampling
from datasets.sampling import make2Dcoords

class ImageSignal(Dataset):
    def __init__(self, data, 
                        width,
                        height,
                        coordinates=None,
                        channels=1,
                        sampling_scheme=Sampling.REGULAR,
                        batch_pixels_perc=None,
                        useattributes=False,
                        attributes={}):
        self.image_t = data
        self.batch_pixels_perc = batch_pixels_perc

        self.data = torch.flatten(data)
        self._width = width
        self._height = height
        self.image_size = width * height
        if batch_pixels_perc is None:
            self.batch_pixels = int(self.image_size)
        else:
            self.batch_pixels = int(batch_pixels_perc*self.image_size)
        if coordinates is None:
            self.coordinates = make2Dcoords(width, height)
        else:
            self.coordinates = coordinates
        self.channels = channels
        self.sampling_scheme = sampling_scheme

        self._useattributes = useattributes

        if attributes:
            self._useattributes = True
            self.d0_mask = attributes.get('d0_mask', None)
            self.d1 = attributes.get('d1', None)
            self.d1_mask = attributes.get('d1_mask', None)
        else:
            self.compute_attributes()

    def init_fromfile(imagepath, useattributes=False,batch_pixels_perc=None,width=None,height=None):
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
                            useattributes=useattributes,
                            batch_pixels_perc=batch_pixels_perc)

    def compute_attributes(self):
        self.d0_mask = torch.ones_like(self.data, dtype=torch.bool)
        # Compute gradient  
        img = self.data.unflatten(0, (self._width, self._height))
        grads_x = scipy.ndimage.sobel(img.numpy(), axis=0)[..., None]
        grads_y = scipy.ndimage.sobel(img.numpy(), axis=1)[..., None]
        grads_x, grads_y = torch.from_numpy(grads_x), torch.from_numpy(grads_y)
        self.d1 = torch.stack((grads_x, grads_y), dim=-1).view(-1, 2)
        self.d1_mask = torch.ones_like(self.d1, dtype=torch.bool)
    
    def drop_attributes(self):
        self._useattributes = False
        self.d1 = None

    def dimensions(self):
        return self._width, self._height

    def image_pil(self):
        return to_pil_image(self.image_t)

    def image_tensor(self):
        return self.data.unsqueeze(0).unflatten(-1, (self._width, self._height))

    def __sub__(self,other):
        data_self = self.data
        data_other = other.data
        subtract_data = data_self - data_other
        width,height = self.dimensions()
        return ImageSignal(subtract_data,
                    width,
                    height,
                    None,
                    self.channels,
                    batch_pixels_perc = self.batch_pixels_perc,
                    useattributes=self._useattributes)
                    
    def __len__(self):
        return self.image_size // self.batch_pixels

    def __getitem__(self, idx):
        if self.batch_pixels == self.image_size:

            in_dict = {'coords': self.coordinates}
            gt_dict = {'d0': self.data.view(-1,1),
                        'd1': self.d1.view(-1,1),
                        'd0_mask': self.d0_mask.view(-1,1),
                        'd1_mask': self.d1_mask.view(-1,1),
                    }
            return  (in_dict, gt_dict)
        else:
            # rand_idcs = np.random.choice(self.image_size, size=self.batch_pixels, replace=True)
            rand_idcs = torch.randint(self.image_size, size=(1, self.batch_pixels))
            rand_coords = self.coordinates[rand_idcs, :]
            d0 = self.data.view(-1,1)
            rand_d0 = d0[rand_idcs, :]
            d0_mask = self.d0_mask.view(-1,1)
            rand_d0_mask = d0_mask[rand_idcs, :]
            d1 = self.data.view(-1,1)
            rand_d1 = d1[rand_idcs, :]
            d1_mask = self.d1_mask.view(-1,1)
            rand_d1_mask = d1_mask[rand_idcs, :]

            in_dict = {'idx':idx,'coords':rand_coords}
            gt_dict = {'d0': rand_d0, 'd1': rand_d1, 'd0_mask': rand_d0_mask, 'd1_mask': rand_d1_mask}

            return  (in_dict,gt_dict)

# OBS: in the future consider to replace the stored self.data with tensor format self.data.view(-1,1)
#      (the same for all attributes, i.e. d1, etc...)
