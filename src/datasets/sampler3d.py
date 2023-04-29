from typing import Sequence, Union
import torch
import scipy
import numpy as np
from torch.utils.data import SequentialSampler, BatchSampler
import torchvision.transforms as T
from .poisson_disc import PoissonDisc

from .constants import Sampling

def make3Dcoords(width:int, height:int, depth:int,
                 start:Union[float, Sequence[float]], 
                 end:Union[float, Sequence[float]]):
    if isinstance(start, Sequence):
        sx, sy, sz = start
    else:
        sx = sy = sz = start
    if isinstance(end, Sequence):
        ex, ey, ez = start
    else:
        ex = ey = ez = end
    lx = torch.linspace(sx, ex, steps=width)
    ly = torch.linspace(sy, ey, steps=height)
    lz = torch.linspace(sz, ez, steps=depth)
    xs, ys, zs = torch.meshgrid(lx, ly, lz, indexing='ij')
    return torch.stack([xs, ys, zs], -1).view(-1, 3)


class RegularSampler3D:

    def __init__(self, img_data, attributes=[]):   
        self.data = img_data
        self.attributes = attributes
        self.key_group = 'c0'

    def get_tuple_dicts(self, sel_idxs):
        coords_sel = self.coords[sel_idxs]
        img_data_sel = self.img_data[sel_idxs]
            
        in_dict = {'coords': coords_sel, 'idx':sel_idxs}
        out_dict = {'d0': img_data_sel.view(-1, 1)}

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

    def make_samples(self, data, width, height, depth, domain, batch_pixel_perc, domain_mask=None):
        self.img_data = torch.flatten(data)
        self.img_width = width
        self.img_height = height
        self.depth = depth
        self.coords = make3Dcoords(width, height, depth, *domain)
        
        self.coords_vis = self.coords

        self.batch_index_dict = {}
        
        if domain_mask is None:
            self.size = len(self.coords)
            self.total_idx_sample = torch.randperm(self.size)
        else:
            # TODO: permute
            self.total_idx_sample = torch.tensor(range(len(self.coords)))[domain_mask]
            self.size = len(self.total_idx_sample)
        
        batch_index = self.create_batch_samples(batch_pixel_perc, self.total_idx_sample)

        self.list_batches = self.create_batches(batch_index)

    def get_samples(self, idx):
        return self.list_batches[idx]


