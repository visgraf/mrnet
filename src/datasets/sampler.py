import torch
import scipy
import numpy as np
from torch.utils.data import SequentialSampler, BatchSampler
import torchvision.transforms as T

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
        print(f'Num of samples: {self.size}')

        self.batch_index_dict = {}

        self.total_idx_sample = torch.randperm(self.size)
        
        batch_index = self.create_batch_samples(batch_pixel_perc,self.total_idx_sample)

        self.list_batches = self.create_batches(batch_index)

    def get_samples(self, idx):
        return self.list_batches[idx]


class PoissonDiscSampler(RegularSampler):
    def __init__(self, img_data, attributes = [], 
                                                k = 30, 
                                                r = 0.8):   
        transform_to_pil = T.ToPILImage()
        self.img_orig = transform_to_pil(img_data)

        self.attributes = attributes
        self.key_group = 'c0'

        self.k = k
        self.r = r

        # Cell side length
        self.a = self.r/np.sqrt(2)


    def make_samples(self, data, width, height, batch_pixel_perc):

        self.img_data = torch.flatten(data)
        self.img_width = width
        self.img_height = height

        # Number of cells in the x- and y-directions of the grid
        self.nx, self.ny = int(self.img_width / self.a) + 1, int(self.img_height / self.a) + 1

        # A list of coordinates in the grid of cells
        self.coords_list = [(ix, iy) for ix in range(self.nx) for iy in range(self.ny)]
        # Initilalize the dictionary of cells: each key is a cell's coordinates, the
        # corresponding value is the index of that cell's point's coordinates in the
        # samples list (or None if the cell is empty).
        self.cells = {coords: None for coords in self.coords_list} 

        self.coords = self.create_sample_points_poisson()
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

    def get_cell_coords(self, pt):
        """Get the coordinates of the cell that pt = (x,y) falls in."""

        return int(pt[0] // self.a), int(pt[1] //  self.a)

    def get_neighbours(self, coords):
        """Return the indexes of points in cells neighbouring cell at coords.

        For the cell at coords = (x,y), return the indexes of points in the cells
        with neighbouring coordinates illustrated below: ie those cells that could 
        contain points closer than r.

                                        ooo
                                        ooooo
                                        ooXoo
                                        ooooo
                                        ooo

        """

        dxdy = [(-1,-2),(0,-2),(1,-2),(-2,-1),(-1,-1),(0,-1),(1,-1),(2,-1),
                (-2,0),(-1,0),(1,0),(2,0),(-2,1),(-1,1),(0,1),(1,1),(2,1),
                (-1,2),(0,2),(1,2),(0,0)]
        neighbours = []
        for dx, dy in dxdy:
            neighbour_coords = coords[0] + dx, coords[1] + dy
            if not (0 <= neighbour_coords[0] <  self.nx and
                    0 <= neighbour_coords[1] <  self.ny):
                # We're off the grid: no neighbours here.
                continue
            neighbour_cell =  self.cells[neighbour_coords]
            if neighbour_cell is not None:
                # This cell is occupied: store this index of the contained point.
                neighbours.append(neighbour_cell)
        return neighbours

    def point_valid(self, pt):
        """Is pt a valid point to emit as a sample?

        It must be no closer than r from any other point: check the cells in its
        immediate neighbourhood.

        """

        cell_coords = self.get_cell_coords(pt)
        for idx in self.get_neighbours(cell_coords):
            nearby_pt = self.samples[idx]
            # Squared distance between or candidate point, pt, and this nearby_pt.
            distance2 = (nearby_pt[0]-pt[0])**2 + (nearby_pt[1]-pt[1])**2
            if distance2 < self.r**2:
                # The points are too close, so pt is not a candidate.
                return False
        # All points tested: if we're here, pt is valid
        return True

    def get_point(self, k, refpt):
        """Try to find a candidate point relative to refpt to emit in the sample.

        We draw up to k points from the annulus of inner radius r, outer radius 2r
        around the reference point, refpt. If none of them are suitable (because
        they're too close to existing points in the sample), return False.
        Otherwise, return the pt.

        """
        i = 0
        while i < k:
            i += 1
            rho = np.sqrt(np.random.uniform(self.r**2, 4 * self.r**2))
            theta = np.random.uniform(0, 2*np.pi)
            pt = refpt[0] + rho*np.cos(theta), refpt[1] + rho*np.sin(theta)
            if not (0 <= pt[0] < self.img_width and 0 <= pt[1] < self.img_height):
                # This point falls outside the domain, so try again.
                continue
            if self.point_valid(pt):
                return pt

        # We failed to find a suitable point in the vicinity of refpt.
        return False

    def create_sample_points_poisson(self):
        # Pick a random point to start with.
        pt = (np.random.uniform(0, self.img_width), np.random.uniform(0, self.img_height))
        self.samples = [pt]
        # Our first sample is indexed at 0 in the samples list...
        self.cells[self.get_cell_coords(pt)] = 0
        # ... and it is active, in the sense that we're going to look for more points
        # in its neighbourhood.
        active = [0]

        nsamples = 1
        # As long as there are points in the active list, keep trying to find samples.
        while active:
            # choose a random "reference" point from the active list.
            idx = np.random.choice(active)
            refpt = self.samples[idx]
            # Try to pick a new point rela  tive to the reference point.
            pt = self.get_point(self.k, refpt)
            if pt:
                # Point pt is valid: add it to the samples list and mark it as active
                self.samples.append(pt)
                nsamples += 1
                active.append(len(self.samples)-1)
                self.cells[self.get_cell_coords(pt)] = len(self.samples) - 1
            else:
                # We had to give up looking for valid points near refpt, so remove it
                # from the list of "active" points.
                active.remove(idx)

        tensor_samples = torch.tensor(self.samples).float()
        tensor_samples[:,0] = 1 - 2*tensor_samples[:,0]/self.img_width
        tensor_samples[:,1] = 1 - 2*tensor_samples[:,1]/self.img_height

        return tensor_samples 


class StochasticSampler:
    def __init__(self, img_data, attributes=[]):   
        self.img_data = img_data
        self.attributes = attributes

        self.transform_to_pil = T.ToPILImage()
        self.img_orig = self.transform_to_pil(img_data)

        self.perc_of_grads = .7

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
        list_coords = list(coords_sel)

            
        in_dict = {'coords': coords_sel, 'idx':sel_idxs}
        out_dict = {}
            
        if class_points == 'c1':
            img_grad_sel_x = [scipy.ndimage.map_coordinates(self.grads_x_numpy,[[self.img_width*(1 +coord[0].item())/2.]  ,
                                                        [self.img_height*(1 + coord[1].item() )/ 2]] )
                                                        for coord in list_coords]
            
            img_grad_sel_y = [scipy.ndimage.map_coordinates(self.grads_y_numpy,[[self.img_width*(1 +coord[0].item())/2.]  ,
                                                        [self.img_height*(1 + coord[1].item() )/ 2]] )
                                                        for coord in list_coords]
            
            img_grad_sel = torch.stack((torch.tensor(img_grad_sel_x,dtype=torch.float), torch.tensor(img_grad_sel_y,dtype=torch.float)), dim=-1).view(-1, 2)
            out_dict['d1'] = img_grad_sel.view(-1,1)

           # print(out_dict['d1'][:6])
            #print(in_dict['coords'][:6])
            #exit(0)
        
        elif class_points == 'c0':
            img_data_sel = [self.img_orig.getpixel( (  self.img_height*(1 +coord[1].item())/2  ,
                                                        self.img_width*(1 + coord[0].item() )/ 2) )/255.
                                                        for coord in list_coords]
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
        self.coords['c0'] = make2Dcoords(width, height, start=-0.98, end=0.98)
        self.coords['c1'] = make2Dcoords(int(width*self.perc_of_grads), int(height*self.perc_of_grads), start=-0.98, end=0.98)

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

    elif sampling_type==Sampling.STOCHASTIC:
        return StochasticSampler(data_to_sample, attributes)

    elif sampling_type==Sampling.POISSON_DISC:
        return PoissonDiscSampler(data_to_sample, attributes)
    


