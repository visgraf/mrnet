import torch
import scipy
import numpy as np
from torch.utils.data import SequentialSampler, BatchSampler
import torchvision.transforms as T
from .poisson_disc import PoissonDisc

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

        self.coords_vis = self.coords

        self.batch_index_dict = {}

        self.total_idx_sample = torch.randperm(self.size)
        
        batch_index = self.create_batch_samples(batch_pixel_perc,self.total_idx_sample)

        self.list_batches = self.create_batches(batch_index)

    def get_samples(self, idx):
        return self.list_batches[idx]


class PoissonDiscSampler(RegularSampler):
    def __init__(self, img_data, attributes = [], 
                                                k = 30, 
                                                r = 0.5):   
        transform_to_pil = T.ToPILImage()
        self.img_orig = transform_to_pil(img_data)

        self.attributes = attributes
        self.key_group = 'c0'

        self.k = k
        self.r = r

    def make_samples(self, data, width, height, batch_pixel_perc):

        self.img_data = torch.flatten(data)
        self.img_width = width
        self.img_height = height

        self.sampler_poison = PoissonDisc(width, height, self.r, self.k)
        self.coords = self.sampler_poison.sample()
        self.coords_vis = make2Dcoords(width, height)

        self.size = len(self.coords)
        print(f'Num of samples: {self.size}')

        self.batch_index_dict = {}
        self.total_idx_sample = torch.randperm(self.size)
        
        batch_index = self.create_batch_samples(batch_pixel_perc,self.total_idx_sample)

        self.list_batches = self.create_batches(batch_index)

    def get_tuple_dicts(self,sel_idxs):
        coords_sel = self.coords[sel_idxs]
        list_coords = list(coords_sel)

        img_data_sel = [self.img_orig.getpixel( (  self.img_height*(1 +coord[1].item())/2  ,
                                                    self.img_width*(1 + coord[0].item() )/ 2) )/255.
                                                    for coord in list_coords]
        img_data_sel = torch.tensor(img_data_sel, dtype = torch.float)
            
        in_dict = {'coords': coords_sel, 'idx':sel_idxs}
        out_dict = {'d0': img_data_sel.view(-1,1)}

        samples = {self.key_group:(in_dict, out_dict)}

        return samples 


class StratifiedSampler:
    def __init__(self, img_data, attributes=[],
                                            k_d0 = 30, 
                                            r_d0 = 0.5,
                                            k_d1 = 30, 
                                            r_d1 = 0.5):   
        self.img_data = img_data
        self.attributes = attributes

        self.transform_to_pil = T.ToPILImage()
        self.img_orig = self.transform_to_pil(img_data)

        self.k_d0 = k_d0
        self.k_d1 = k_d1

        self.r_d0 = r_d0
        self.r_d1 = r_d1

    def compute_gradients(self):
        img = self.img_data.unflatten(0, (self.img_width, self.img_height))

        grads_x = scipy.ndimage.sobel(img, axis=0)[..., None]
        grads_y = scipy.ndimage.sobel(img, axis=1)[..., None]

        grads_x, grads_y = torch.from_numpy(grads_x), torch.from_numpy(grads_y)
        self.img_grad = torch.stack((grads_x, grads_y), dim=-1).view(-1, 2)

        self.grads_x_numpy, self.grads_y_numpy = np.copy(grads_x.squeeze()), np.copy(grads_y.squeeze())

    def get_tuple_dicts(self,sel_idxs, class_points):
        coords_type_points = self.coords[class_points]
        coords_sel = coords_type_points[sel_idxs]
            
        in_dict = {'coords': coords_sel, 'idx':sel_idxs}
        out_dict = {}
            
        if class_points == 'c1':

            img_grad_sel_x = np.array([scipy.ndimage.map_coordinates(self.grads_x_numpy,[[self.img_width*(1 +coord[0].item())/2.]  ,
                                                        [self.img_height*(1 + coord[1].item() )/ 2]] )
                                                        for coord in coords_sel])
            
            img_grad_sel_y = np.array([scipy.ndimage.map_coordinates(self.grads_y_numpy,[[self.img_width*(1 +coord[0].item())/2.]  ,
                                                        [self.img_height*(1 + coord[1].item() )/ 2]] )
                                                        for coord in coords_sel])
            
            img_grad_sel = torch.stack( [ torch.tensor(img_grad_sel_x,dtype=torch.float), torch.tensor(img_grad_sel_y,dtype=torch.float)], dim=-1)
            out_dict['d1'] = img_grad_sel.view(-1,1)
        
        elif class_points == 'c0':
            img_data_sel = [self.img_orig.getpixel( (  self.img_height*(1 +coord[1].item())/2  ,
                                                        self.img_width*(1 + coord[0].item() )/ 2) )/255.
                                                        for coord in coords_sel]
            img_data_sel = torch.tensor(img_data_sel, dtype = torch.float)
            out_dict = {'d0': img_data_sel.view(-1,1)}
        
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

        self.coords = {}
        self.coords['c0'] = PoissonDisc(width, height, self.r_d0, self.k_d0).sample()
        self.coords['c1'] = PoissonDisc(width, height, self.r_d1, self.k_d1).sample()
        self.coords_vis = make2Dcoords(width, height)

        if 'd1' in self.attributes:
            self.compute_gradients()
        
        self.total_idx_sample = {}
        self.total_idx_sample['c0'] = list(range(len(self.coords['c0'])))
        self.total_idx_sample['c1'] = list(range(len(self.coords['c1'])))

        self.total_ids_in_batch = {}
        self.total_ids_in_batch['c0'] = int(len(self.coords['c0']))
        self.total_ids_in_batch['c1'] = int(len(self.coords['c1']))
        
        self.batch_index_dict = {}
        self.batch_index_dict['c0'] = self.create_batch_samples(batch_pixel_perc,self.total_idx_sample['c0'])
        self.batch_index_dict['c1'] = self.create_batch_samples(batch_pixel_perc,self.total_idx_sample['c1'])


        self.batch_dict = {}
        self.batch_dict['c0'] = self.create_batches(self.batch_index_dict['c0'],'c0')
        self.batch_dict['c1'] = self.create_batches(self.batch_index_dict['c1'],'c1')

        self.list_batches = [{'c0':self.batch_dict['c0'][i], 'c1':self.batch_dict['c1'][i] } for i in range(self.total_size())]

    def get_samples(self, idx):
        return self.list_batches[idx]

    
    
def samplerFactory(sampling_type:Sampling, data_to_sample, attributes):
    if sampling_type==Sampling.REGULAR:
        return RegularSampler(data_to_sample, attributes)

    elif sampling_type==Sampling.STRATIFIED:
        return StratifiedSampler(data_to_sample, attributes)

    elif sampling_type==Sampling.POISSON_DISC:
        return PoissonDiscSampler(data_to_sample, attributes)
    


