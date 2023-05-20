import numpy as np
from PIL import Image
from noise import snoise3
from typing import Sequence
from scipy.spatial import cKDTree
import torch


def make_grid_coords(nsamples, start, end, dim):
    if not isinstance(nsamples, Sequence):
        nsamples = dim * [nsamples]
    if not isinstance(start, Sequence):
        start = dim * [start]
    if not isinstance(end, Sequence):
        end = dim * [end]
    if len(nsamples) != dim or len(start) != dim or len(end) != dim:
        raise ValueError("'nsamples'; 'start'; and 'end' should be a single value or have same  length as 'dim'")
    
    dir_samples = tuple([np.linspace(start[i], end[i], nsamples[i]) 
                   for i in range(dim)])
    grid = np.stack(np.meshgrid(*dir_samples, indexing='ij'), axis=-1)
    return grid.reshape(-1, dim)

vnoise3 = np.vectorize(snoise3)

def turbulence(p, pixelsize):
    t = np.zeros(len(p))
    scale = 1
    while (scale > pixelsize):
        v = p / scale
        t += vnoise3(v[..., 0], v[..., 1], v[..., 2]) * scale
        scale /= 2 
    return t


def colorful(values):
    colors = np.ones((len(values), 3)) * 0.8
    colors[(0.25 < values) & (values < 0.50)] = np.array([1.0, 0.2, 0.2])
    colors[(0.50 < values) & (values < 0.75)] = np.array([0.2, 1.0, 0.2])
    colors[values > 0.75] = np.array([0.2, 0.2, 1.0])

    return colors

def colorful_texture(size):
    grid = make_grid_coords(size, -1, 1, 3)
    print(grid.shape)
    vfunc = np.vectorize(snoise3)
    # color = np.ones((len(grid), 3)) * np.abs(vfunc(grid[:, 0:1], grid[:, 1:2], grid[:, 2:]))
    k = 1.4
    color = colorful(np.abs(vfunc(k * grid[:, 0], k * grid[:, 1], k * grid[:, 2])))
    texture = (color.reshape((size, size, size, 3)) * 255).astype(np.uint8)
    print(texture.shape)
    return texture

def marble_texture(pixelsize=1/64, color_map=None):
    if color_map is None:
        color_map = lambda k: marble_color(torch.sin(k * torch.pi))
    def procedure(point):
        x = point[..., 0] + turbulence(point, pixelsize)
        # return marble_color(np.sin(x * np.pi))
        return color_map(x)
    return procedure

def marble_color(values):
    rgb = torch.zeros((len(values), 3))
    x = torch.sqrt(values + 1) * .7071
    rgb[..., 1] = .3 + .8 * x
    x = torch.sqrt(x)
    rgb[..., 0] = .3 + .6 * x
    rgb[..., 2] = .6 + .4 * x
    return torch.abs(rgb).clamp(0, 1)

def voronoi_texture(ncells, domain=(-1, 1), colors_map=[], line_tol=4e-3):
    points = (torch.rand((ncells, 3)).numpy() * (domain[1] - domain[0])) + domain[0]
    voronoi_kdtree = cKDTree(points)
    if not colors_map:
        colors_map = torch.rand((ncells//2, 3))
    def colorize(x):
        dist, regions = voronoi_kdtree.query(x, k=2)
        colors = colors_map[regions[..., 0] % len(colors_map)]
        mask = abs(dist[..., 0] - dist[..., 1]) < line_tol
        colors[mask] = torch.zeros(3)
        return colors
    return colorize


def save_texture(texture, prefix, size):
    Image.fromarray(texture[:, :, 7, :], mode='RGB').save(f'img/{prefix}_tex{size}xy.png')
    Image.fromarray(texture[:, 7, :, :], mode='RGB').save(f'img/{prefix}_tex{size}xz.png')
    Image.fromarray(texture[7, :, :, :], mode='RGB').save(f'img/{prefix}_tex{size}yz.png')
    texture = texture.astype(np.float32) / 255
    np.save(f'voxels/{prefix}_noise3d{size}.npy', texture)

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

def perlin_noise(samples, scale=10, octaves=1, p=1):
    pnoise = 0
    for i in range(octaves):
        partial = noise(2**i * scale, samples)/(p**i)
        pnoise = partial + pnoise
    return pnoise

def checker(texsize, cubesize):
    # Create a 3D grid of coordinates
    x, y, z = np.mgrid[0:texsize, 0:texsize, 0:texsize]

    # Generate a sine wave pattern in each dimension
    sine_x = np.sin(2 * np.pi * x / cubesize)
    sine_y = np.sin(2 * np.pi * y / cubesize)
    sine_z = np.sin(2 * np.pi * z / cubesize)

    # Combine the sine waves and threshold to create checkerboard pattern
    return ((sine_x + sine_y + sine_z) > 0).astype(np.float32)


def solid_texture(N, noise_scale=0.1):
    x, y, z = np.ogrid[-1:1:N*1j, -1:1:N*1j, -1:1:N*1j]
    r = np.sqrt(x**2 + y**2 + z**2)
    solid = np.zeros_like(r)
    solid[r < 0.5] = 1
    noise = noise_scale * np.random.randn(N, N, N)
    solid += noise
    solid[solid < 0] = 0
    solid[solid > 1] = 1
    return solid.astype(np.float32)



if __name__ == '__main__':
    from utils import make_domain_slices
    # def test_marble_v2(res, views=['x', 'y', 'z', 'xy'], cmap=None):
    #     proc = marble_v2(cmap)
    #     slices = make_domain_slices(res, -1, 1, views)
    #     for i, view in enumerate(views):
    #         values = proc(slices[i].view(-1, 3)).view(res, res, 3)
    #         values = (values.clamp(0, 1).numpy() * 255).astype(np.uint8)
    #         Image.fromarray(values).save(f"img/temp/mv2_{view}.png")
    
    print("DONE")
