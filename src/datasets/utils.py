import torch 
import numpy as np
# Adapted from https://github.com/makeyourownalgorithmicart/makeyourownalgorithmicart/blob/master/blog/perlin_gradient_noise/1d_perlin.ipynb

def blend(x):
  return 6*x**5 - 15*x**4 + 10*x**3

def noise(scale, samples):
    # create a list of 2d vectors
    angles = torch.rand(scale) * 2*np.pi
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)

    x = torch.linspace(0, scale-1, samples)
    noise_values = []
    for value in x[:-1]:
        i = int(np.floor(value))
        lower, upper = gradients[i], gradients[i+1]

        dot1 = torch.dot(lower, torch.tensor([value - i, 0]))
        dot2 = torch.dot(upper, torch.tensor([value - i - 1, 0]))
        # TODO: review interpolation
        k1 = blend(value-i)
        k2 = blend(i+1 - value)
        interpolated = k1*dot2 + k2*dot1 
        noise_values.append(interpolated.item())
    
    noise_values.append(0.0)
    return torch.tensor(noise_values)

def perlin_noise(samples, scale=10, octaves=1, p=1, batch_size=0):
    pnoise = 0
    for i in range(octaves):
        partial = noise(2**i * scale, samples)/(p**i)
        pnoise = partial + pnoise
    return pnoise