import numpy as np
import torch
import os

'''
Used the code from https://github.com/scipython/scipython-maths
MIT License

Copyright (c) 2021 Christian Hill

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

'''
from scipy.stats import qmc
import math


class PoissonDisc():
    def __init__(self, dimensions):

        width = min(dimensions)
        self.tensor_Len = len(dimensions)
        self.radius = 1. / ((width - 1))

    def sample_points(self):
        self.engine = qmc.PoissonDisk(d=self.tensor_Len, radius=self.radius)
        self.samples = self.engine.fill_space()

        return self.samples

    def sample(self):
        string_name = 'poisson_cache/poisson_radius_{r:.10f}.npy'.format(
            r=self.radius)
        if os.path.exists(string_name):
            print("Loading poisson points from cache")
            self.samples = np.load(string_name)
        else:
            print("We will compute the poisson points")
            self.samples = self.sample_points()
            np.save(string_name, self.samples)

        tensor_samples = torch.tensor(self.samples).float()
        tensor_samples = 2 * tensor_samples - 1

        return tensor_samples
