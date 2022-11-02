import torch
import scipy
import numpy as np
from torch.utils.data import SequentialSampler, BatchSampler

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
        self.key_group = 'c0'

    def get_tuple_dicts(self,sel_idxs):
        coords_sel = self.coords[sel_idxs]
        img_data_sel = self.img_data[sel_idxs]
            
        in_dict = {'coords': coords_sel, 'idx':sel_idxs}
        out_dict = {'d0': img_data_sel.view(-1,1)}

        samples = {self.key_group:(in_dict, out_dict)}

        return samples

    def create_batch_samples(self, batch_pixel_perc, indices_to_sample):
        batch_size = int(len(indices_to_sample)*batch_pixel_perc)

        return (list(BatchSampler(indices_to_sample, batch_size=batch_size, drop_last=False)))

    def total_size(self):
        return len(self.list_batches)

    
    def create_batches(self, batch_index_samples):
        list_samples = []

        for sel_idxs in batch_index_samples:
            samples = self.get_tuple_dicts(sel_idxs)
            list_samples.append(samples)
        
        return list_samples

    def make_samples(self, data, width, height, batch_pixel_perc):
        self.img_data = torch.flatten(data)
        self.img_width = width
        self.img_height = height
        self.size = width*height
        self.coords = make2Dcoords(width, height)

        self.batch_index_dict = {}

        self.total_idx_sample = torch.randperm(self.size)
        
        batch_index = self.create_batch_samples(batch_pixel_perc,self.total_idx_sample)

        self.list_batches = self.create_batches(batch_index)

    def get_samples(self, idx):
        return self.list_batches[idx]
        


class StochasticSampler:
    def __init__(self, img_data, attributes=[]):   
        self.img_data = img_data
        self.attributes = attributes

        self.perc_of_no_grads = .9

    def compute_gradients(self):
        img = self.img_data.unflatten(0, (self.img_width, self.img_height))
        grads_x = scipy.ndimage.sobel(img.numpy(), axis=0)[..., None]
        grads_y = scipy.ndimage.sobel(img.numpy(), axis=1)[..., None]
        grads_x, grads_y = torch.from_numpy(grads_x), torch.from_numpy(grads_y)
        self.img_grad = torch.stack((grads_x, grads_y), dim=-1).view(-1, 2)

    def get_tuple_dicts(self,sel_idxs, class_points):
        coords_sel = self.coords[sel_idxs]
        img_data_sel = self.img_data[sel_idxs]
            
        in_dict = {'coords': coords_sel, 'idx':sel_idxs}
        out_dict = {'d0': img_data_sel.view(-1,1)}
            
        if class_points == 'c1':
            img_grad_sel = self.img_grad[sel_idxs]
            out_dict['d1'] = img_grad_sel.view(-1,1)

        samples = (in_dict, out_dict)

        return samples

    def create_batch_samples(self, batch_pixel_perc, indices_to_sample):
        batch_size = int(len(indices_to_sample)*batch_pixel_perc)

        return (list(BatchSampler(indices_to_sample, batch_size=batch_size, drop_last=False)))

    def create_batches(self, batch_index_samples, class_points):
        list_samples = []

        for sel_idxs in batch_index_samples:
            samples = self.get_tuple_dicts(sel_idxs, class_points)
            list_samples.append(samples)
        
        return list_samples

    def total_size(self):
        return len(self.batch_index_dict['c0'])

    def make_samples(self, data, width, height, batch_pixel_perc):
        self.img_data = torch.flatten(data)
        self.img_width = width
        self.img_height = height
        self.size = width*height
        self.coords = make2Dcoords(width, height)

        if 'd1' in self.attributes:
            self.compute_gradients()

        self.batch_dict = {}
        self.batch_index_dict = {}

        self.total_idx_sample = list(range(self.size))
        self.total_ids_in_batch = int(self.size*self.perc_of_no_grads)
        
        self.batch_index_dict['c0'] = self.create_batch_samples(batch_pixel_perc,self.total_idx_sample[self.total_ids_in_batch:])
        self.batch_index_dict['c1'] = self.create_batch_samples(batch_pixel_perc,self.total_idx_sample[:self.total_ids_in_batch])


        self.batch_dict['c0'] = self.create_batches(self.batch_index_dict['c0'],'c0')
        self.batch_dict['c1'] = self.create_batches(self.batch_index_dict['c1'],'c1')

        self.list_batches = [{'c0':self.batch_dict['c0'][i], 'c1':self.batch_dict['c1'][i] } for i in range(self.total_size())]

    def get_samples(self, idx):
        return self.list_batches[idx]

    
    
def samplerFactory(sampling_type:Sampling, data_to_sample, attributes):
    if sampling_type==Sampling.REGULAR:
        return RegularSampler(data_to_sample, attributes)

    elif sampling_type==Sampling.STOCHASTIC:
        return StochasticSampler(data_to_sample, attributes)
    


