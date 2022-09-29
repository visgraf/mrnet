import torch
import scipy
import numpy as np

from .constants import Sampling

def make2Dcoords(width, height, start=-1, end=1):
    lx = torch.linspace(start, end, steps=width)
    ly = torch.linspace(start, end, steps=height)
    xs, ys = torch.meshgrid(lx, ly, indexing='ij')
    return torch.stack([xs, ys], -1).view(-1, 2)

class Sampler:
    def __init__(self) -> None:
        pass

    def make_samples(self, image):
        pass

    def total_size(self):
        pass

    def get_samples(self, idx):
        pass

class RegularSampler(Sampler):

    def __init__(self, img_data, attributes=[]):   
        self.img_data = img_data
        self.attributes = attributes

    def compute_gradients(self):
        img = self.img_data.unflatten(0, (self.img_width, self.img_height))
        grads_x = scipy.ndimage.sobel(img.numpy(), axis=0)[..., None]
        grads_y = scipy.ndimage.sobel(img.numpy(), axis=1)[..., None]
        grads_x, grads_y = torch.from_numpy(grads_x), torch.from_numpy(grads_y)
        self.img_grad = torch.stack((grads_x, grads_y), dim=-1).view(-1, 2)

    def make_samples(self, data, width, height):
        self.img_data = torch.flatten(data)
        self.img_width = width
        self.img_height = height
        self.coords = make2Dcoords(width, height)
        
        if 'd1' in self.attributes:
            self.compute_gradients()

    def total_size(self):
        return self.img_data.size() 

    def get_samples(self, idx, batch_pixel_perc):
        num_of_elements=self.img_data.shape[0]
        rand_idcs = np.random.choice(num_of_elements , size=int(batch_pixel_perc*num_of_elements))

        coords_sel = self.coords[rand_idcs]
        img_data_sel = self.img_data[rand_idcs]
        
        in_dict = {'ídx': idx, 'coords': coords_sel}
        out_dict = {'d0': img_data_sel.view(-1,1)}
        
        if 'd1' in self.attributes:
            img_grad_sel = self.img_grad[rand_idcs]
            out_dict['d1'] = img_grad_sel.view(-1,1)

        samples = (in_dict, out_dict)
        return samples


def samplerFactory(sampling_type:Sampling, data_to_sample, attributes):
    if sampling_type==Sampling.REGULAR:
        return RegularSampler(data_to_sample, attributes)


