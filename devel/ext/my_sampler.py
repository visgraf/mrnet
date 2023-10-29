import torch
from mrnet.ext.stochastic_samplers import StochasticSampler

    
class MyUniformSampler(StochasticSampler):
    def stochastic_sample_method(self, coords_orig):
        random_samples = torch.rand_like(coords_orig)

        random_samples = 1 - 2*random_samples

        return random_samples    