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
        t += vnoise3(v[:, 0], v[:, 1], v[:, 2]) * scale
        scale /= 2 
    return t

# def marble_texture(size, pixelsize=1/64):
#     grid = make_grid_coords(size, -1, 1, 3)
#     x = grid[:, 0] + turbulence(grid, pixelsize)
#     colors = marble_color(np.sin(x * np.pi))
#     return (colors.reshape((size, size, size, 3)) * 255).astype(np.uint8)

# def marble_color(values):
#     rgb = np.zeros((len(values), 3))
#     # x = np.sin((pos(2) + 3.0 * values) * np.pi)
#     x = np.sqrt(values + 1) * .7071
#     rgb[:, 2] = .3 + .8 * x
#     x = np.sqrt(x)
#     rgb[:, 1] = .3 + .6 * x
#     rgb[:, 0] = .6 + .4 * x
#     return np.clip(np.abs(rgb), 0, 1)

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
        x = point[:, 0] + turbulence(point, pixelsize)
        # return marble_color(np.sin(x * np.pi))
        return color_map(x)
    return procedure

def marble_color(values):
    rgb = torch.zeros((len(values), 3))
    x = torch.sqrt(values + 1) * .7071
    rgb[:, 1] = .3 + .8 * x
    x = torch.sqrt(x)
    rgb[:, 0] = .3 + .6 * x
    rgb[:, 2] = .6 + .4 * x
    return torch.abs(rgb).clamp(0, 1)

def voronoi_texture(ncells, domain=(-1, 1), colors_map=[], line_tol=4e-3):
    points = (torch.rand((ncells, 3)).numpy() * (domain[1] - domain[0])) + domain[0]
    voronoi_kdtree = cKDTree(points)
    if not colors_map:
        colors_map = torch.rand((ncells//2, 3))
    def colorize(x):
        dist, regions = voronoi_kdtree.query(x, k=2)
        colors = colors_map[regions[:, 0] % len(colors_map)]
        mask = abs(dist[:, 0] - dist[:, 1]) < line_tol
        colors[mask] = torch.zeros(3)
        return colors
    return colorize


def save_texture(texture, prefix, size):
    Image.fromarray(texture[:, :, 7, :], mode='RGB').save(f'img/{prefix}_tex{size}xy.png')
    Image.fromarray(texture[:, 7, :, :], mode='RGB').save(f'img/{prefix}_tex{size}xz.png')
    Image.fromarray(texture[7, :, :, :], mode='RGB').save(f'img/{prefix}_tex{size}yz.png')
    texture = texture.astype(np.float32) / 255
    np.save(f'voxels/{prefix}_noise3d{size}.npy', texture)

if __name__ == '__main__':
    # Define the parameters for the Perlin noise function
    scale = 50.0  # The scale of the noise
    octaves = 10   # The number of octaves in the noise
    persistence = 0.6   # The persistence of the noise
    lacunarity = 3.0     # The lacunarity of the noise

    # Save the texture as an image file
    size = 128
    name = "marble"
    # texture = colorful_texture(size)
    texture = marble_texture(size)
    save_texture(texture, name, size)
    print("DONE")
