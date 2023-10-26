import torch
from .sift_sampler_code_2d import get_samples_sift
from mrnet.ext.stochastic_samplers import StochasticSampler
from mrnet.ext.poisson_disc import PoissonDisc


class AdaptiveSamplerSIFT(StochasticSampler):
    def stochastic_sample_method(self, coords_orig):

        sift_points = get_samples_sift(self.data.numpy())
        sift_points = torch.tensor(sift_points).float()

        poisson_disc = PoissonDisc(self.data_shape())
        coords_poisson = poisson_disc.sample()

        points_sampled = torch.cat([coords_poisson, sift_points])

        return points_sampled
