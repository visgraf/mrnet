import torch
from torch.utils.data import BatchSampler
from .poisson_disc import PoissonDisc
import random
from scipy.interpolate import griddata
from typing import Sequence


from mrnet.datasets.sampler import Sampler
import numpy as np


def make_grid_coords(nsamples, start, end, dim, flatten=True):
    if not isinstance(nsamples, Sequence):
        nsamples = dim * [nsamples]
    if not isinstance(start, Sequence):
        start = dim * [start]
    if not isinstance(end, Sequence):
        end = dim * [end]
    if len(nsamples) != dim or len(start) != dim or len(end) != dim:
        raise ValueError(
            "'nsamples'; 'start'; and 'end' should be a single value or have same  length as 'dim'")

    dir_samples = tuple([torch.linspace(start[i], end[i], steps=nsamples[i])
                         for i in range(dim)])
    grid = torch.stack(torch.meshgrid(*dir_samples, indexing='ij'), dim=-1)
    return grid.reshape(-1, dim) if flatten else grid


def make2Dcoords(width, height, start=-1, end=1):
    lx = torch.linspace(start, end, steps=width)
    ly = torch.linspace(start, end, steps=height)
    xs, ys = torch.meshgrid(lx, ly, indexing='ij')
    return torch.stack([xs, ys], -1).view(-1, 2)


def set_seed(seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_interpolated_values(img_data, coords, orig_coords):

    interpolated_pixel = griddata(orig_coords, img_data, coords)
    return interpolated_pixel


class StochasticSampler(Sampler):
    def make_stochastic_samples():
        pass

    def make_samples(self, domain_mask=None):
        self.key_group = 'c0'
        dimension = len(self.data_shape())
        self._coords_orig = make_grid_coords(self.data_shape(),
                                             *self.domain, dim=dimension)

        self._coords = self.stochastic_sample_method(self._coords_orig)
        self._coords = torch.clamp(self._coords, min=-1, max=1)

        n = len(self._coords)
        if domain_mask is not None:
            sampled_indices = torch.arange(0, n, dtype=torch.long)[
                domain_mask.view(-1)]
            if self.shuffle:
                random_idx = torch.randperm(len(sampled_indices))
                sampled_indices = sampled_indices[random_idx]
        else:
            sampled_indices = (torch.randperm(n) if self.shuffle
                               else torch.arange(0, n, dtype=torch.long))

        index_batches = list(
            BatchSampler(sampled_indices, self.batch_size, drop_last=False)
        )
        flatdata = {'d0': self.data.view(self.data_channels(),
                                         -1).permute((1, 0))}
        nsamples = len(flatdata['d0'])
        for key, value in self._attributes.items():
            flatdata[key] = value.view(nsamples, -1)

        self.batches = [self.get_tuple_dicts(
            torch.Tensor(idx_batch).long(), flatdata)
            for idx_batch in index_batches]

    def get_tuple_dicts(self, sel_idxs, flatdata):
        coords_sel = self._coords[sel_idxs, ...]
        in_dict = {'coords': coords_sel, 'idx': sel_idxs}
        out_dict = {}
        data_shape = self.data.shape

        orig_coordinates_numpy = self._coords_orig.detach().numpy()
        coords_sel_rescaled_numpy = coords_sel.detach().numpy()
        for key in flatdata.keys():
            data_group = flatdata[key]
            reshape_data_group = torch.reshape(data_group, [-1, data_shape[0]])
            reshape_data_group_numpy = reshape_data_group.numpy()

            reshape_data_numpy_group = get_interpolated_values(
                reshape_data_group_numpy, coords_sel_rescaled_numpy, orig_coordinates_numpy)

            out_dict[key] = torch.tensor(reshape_data_numpy_group).float()
        samples = {self.key_group: (in_dict, out_dict)}
        return samples


class JitteredSampler(StochasticSampler):
    def stochastic_sample_method(self, coords_orig):
        random_noise = 2*torch.randn_like(coords_orig) - 1

        for index, dim_range in enumerate(self.data_shape()):
            random_noise[:, index] = 0.5*random_noise[:, index]/(dim_range-1)

        coords_pertubed = coords_orig + random_noise

        return coords_pertubed


class PoissonDiscSampler(StochasticSampler):
    def stochastic_sample_method(self, coords_orig):

        poisson_disc = PoissonDisc(self.data_shape())
        coords_poisson = poisson_disc.sample()
        return coords_poisson


class UniformSampler(StochasticSampler):
    def stochastic_sample_method(self, coords_orig):
        random_samples = torch.rand_like(coords_orig)

        random_samples = 1 - 2*random_samples

        return random_samples
