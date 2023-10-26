import numpy as np
import torch
import os


from scipy.stats import qmc
import math


class PoissonDisc():
    def __init__(self, dimensions, folder_name='poisson_cache', verbose=False):

        width = min(dimensions)
        self.tensor_Len = len(dimensions)
        self.radius = 1. / ((width - 1))

        self.width = width
        self.folder_name = folder_name
        self.verbose = verbose

    def sample_points(self):
        self.engine = qmc.PoissonDisk(d=self.tensor_Len, radius=self.radius)
        self.samples = self.engine.fill_space()

        return self.samples

    def sample(self):
        string_name = os.path.join(
            self.folder_name, f'poisson_radius_{self.width}.npy')
        if os.path.exists(string_name):

            if self.verbose:
                print("Loading poisson points from cache")
            self.samples = np.load(string_name)
        else:

            print("We will compute the poisson points")
            self.samples = self.sample_points()
            np.save(string_name, self.samples)

        tensor_samples = torch.tensor(self.samples).float()
        tensor_samples = 2 * tensor_samples - 1

        return tensor_samples
