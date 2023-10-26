from typing import Sequence, Union
import torch
import scipy
import numpy as np
from torch.utils.data import BatchSampler
import torchvision.transforms as T
from .utils import make_grid_coords


class Sampler:
    """Base Sampler class"""

    def __init__(self, data, domain, attributes, batch_size, shuffle=False):
        self.data = data
        self.domain = domain
        self._attributes = attributes
        self.batch_size = (batch_size if batch_size > 0
                           else len(torch.flatten(data)))
        self.shuffle = shuffle
        self.batches = []
        self.mask = None

    def __len__(self):
        # lazy initialization
        if not self.batches:
            self.make_samples(self.mask)
        return len(self.batches)

    def __getitem__(self, idx):
        # lazy inialization
        if not self.batches:
            self.make_samples(self.mask)
        return self.batches[idx]

    def add_mask(self, mask):
        self.mask = mask
        if self.batches:
            self.make_samples(self.mask)

    @property
    def coords(self):
        try:
            return self._coords
        except AttributeError:
            self.make_samples(self.mask)
        return self._coords

    @property
    def attributes(self):
        return self._attributes

    @attributes.setter
    def attributes(self, value):
        self._attributes = value
        if self.batches:
            self.make_samples(self.mask)

    def data_channels(self):
        return self.data.shape[0]

    def data_shape(self):
        return self.data.shape[1:]

    def make_samples(self):
        raise NotImplementedError()

    def total_nsamples(self):
        raise NotImplementedError()

    def scheme(self):
        raise NotImplemented()


class RegularSampler(Sampler):

    def make_samples(self, domain_mask=None):
        self.key_group = 'c0'
        dimension = len(self.data_shape())
        self._coords = make_grid_coords(self.data_shape(),
                                        *self.domain, dim=dimension)

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
        # if 'd1' in self._attributes.keys():
        #     flatdata['d1'] = torch.sum(self._attributes['d1'],
        #                                dim=0).view(-1, dimension)
        nsamples = len(flatdata['d0'])
        for key, value in self._attributes.items():
            flatdata[key] = value.view(nsamples, -1)

        self.batches = [self.get_tuple_dicts(
            torch.Tensor(idx_batch).long(), flatdata)
            for idx_batch in index_batches]

    def get_tuple_dicts(self, sel_idxs, flatdata):
        coords_sel = self._coords[sel_idxs]
        in_dict = {'coords': coords_sel, 'idx': sel_idxs}
        out_dict = {}
        for key in flatdata.keys():
            out_dict[key] = flatdata[key][sel_idxs]
        samples = {self.key_group: (in_dict, out_dict)}
        return samples

    def scheme(self):
        return "regular"

# TODO: ponder if it should be excluded,
# making the reflection loss the default solution


class ReflectSampler(RegularSampler):
    def make_samples(self, domain_mask=None):
        self.key_group = 'c0'
        # double the points
        nsamples = tuple(np.array(self.data_shape()) * 2)
        domain = tuple(np.array(self.domain) * 2)
        self._coords = make_grid_coords(nsamples,
                                        *domain,
                                        dim=len(self.data_shape()))

        if domain_mask is None:
            n = len(self._coords)
            sampled_indices = (torch.randperm(n) if self.shuffle
                               else torch.arange(0, n, dtype=torch.long))
        else:
            # TODO: permute; flatten domain_mask?
            sampled_indices = torch.tensor(
                range(len(self._coords)))[domain_mask]

        index_batches = list(
            BatchSampler(sampled_indices, self.batch_size, drop_last=False)
        )

        d0 = self.extend_by_reflection(self.data, (0, 1))
        flatdata = {'d0': d0.view(self.data_channels(), -1).permute((1, 0))}
        if 'd1' in self._attributes.keys():
            d1 = torch.sum(self._attributes['d1'], dim=0).unsqueeze(0)
            d1 = self.extend_by_reflection(d1, (0, 1)).squeeze(0).view(-1, 2)
            flatdata['d1'] = d1
        self.batches = [self.get_tuple_dicts(
            torch.Tensor(idx_batch).long(), flatdata)
            for idx_batch in index_batches]

    def extend_by_reflection(self, data, dims):
        channels = data.shape[0]
        for ch in range(channels):
            reflected = []
            extended = data[ch]
            for dim in dims:
                flipped = extended.flip(dim)
                extd = self.data_shape()[dim] // 2
                if dim == 0:
                    values = [flipped[-extd:, ...],
                              extended,
                              flipped[:extd, ...]]
                elif dim == 1:
                    values = [flipped[:, -extd:, ...],
                              extended,
                              flipped[:, :extd, ...]]
                elif dim == 2:
                    values = [flipped[..., -extd:],
                              extended,
                              flipped[..., :extd]]
                else:
                    raise ValueError("Unsupported dimension")
                extended = torch.cat(values, dim=dim)
            reflected.append(extended)
        reflected = torch.stack(reflected)
        return reflected

    def scheme(self):
        return "reflect"

# TODO: refactor and extend to work with multiple dimensions;
# include in the sampler class hierarchy


class ProceduralSampler:
    def __init__(self, procedure,
                 domain,
                 attributes,
                 batch_size,
                 pseudo_shape) -> None:
        self.procedure = procedure
        self.domain = domain
        self._attributes = attributes

        RANDOM_SEED = 777
        self.rng = np.random.default_rng(RANDOM_SEED)
        self._pseudo_shape = pseudo_shape

        dims = pseudo_shape[1:]
        prod = dims[0] * dims[1] * dims[2]
        self._num_samples = prod // batch_size
        if prod % batch_size != 0:
            self._num_samples += 1
        self.batch_size = (batch_size if batch_size > 0
                           else prod * pseudo_shape[0])

    def __len__(self):
        return self._num_samples

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration
        possible_values = torch.linspace(*self.domain, self.shape[1])
        channels = self.shape[0]
        coords = self.rng.choice(possible_values,
                                 (self.batch_size, 3),
                                 replace=True)
        coords = torch.from_numpy(coords)
        in_dict = {'coords': coords}
        out_dict = {'d0': self.procedure(coords)}
        return {'c0': (in_dict, out_dict)}

    @property
    def shape(self):
        return self._pseudo_shape


SAMPLING_CLASSES = {
    'regular': RegularSampler,
    'reflect': ReflectSampler
}


# class SamplerFactory:
#     subclass = {
#         Sampling.REGULAR: RegularSampler,
#         Sampling.REFLECT: ReflectSampler,
#         Sampling.POISSON_DISC: PoissonDiscSampler,
#         Sampling.STRATIFIED: StratifiedSampler,
#     }
#     def init(sampling_type:Sampling,
#              data, domain,
#              attributes, batch_size, shuffle) -> Sampler:
#         try:
#             SamplerClass = SamplerFactory.subclass[sampling_type]
#         except KeyError:
#             raise ValueError(f"Invalid sampling type {sampling_type}")
#         return SamplerClass(data,
#                             domain,
#                             attributes,
#                             batch_size,
#                             shuffle)
