import torch
import scipy

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

    def __init__(self, img_data):   
        self.img_data = img_data

    def compute_attributes(self):
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
        self.compute_attributes()

    def total_size(self):
        return self.img_data.size() + self.img_grad.size()

    def get_samples(self, idx):
        in_dict = {'ídx': idx, 'coords': self.coords}
        out_dict = {'d0': self.img_data.view(-1,1),
                     'd1': self.img_grad.view(-1,1),
                    }
        samples = (in_dict, out_dict)
        return samples


def samplerFactory(sampling_type:Sampling, data_to_sample):

    if sampling_type==Sampling.REGULAR:
        return RegularSampler(data_to_sample)


