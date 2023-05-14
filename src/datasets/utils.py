import torch 
import numpy as np
from typing import Sequence, Union

# Adapted from https://github.com/makeyourownalgorithmicart/makeyourownalgorithmicart/blob/master/blog/perlin_gradient_noise/1d_perlin.ipynb


def make_grid_coords(nsamples, start, end, dim, flatten=True):
    if not isinstance(nsamples, Sequence):
        nsamples = dim * [nsamples]
    if not isinstance(start, Sequence):
        start = dim * [start]
    if not isinstance(end, Sequence):
        end = dim * [end]
    if len(nsamples) != dim or len(start) != dim or len(end) != dim:
        raise ValueError("'nsamples'; 'start'; and 'end' should be a single value or have same  length as 'dim'")
    
    dir_samples = tuple([torch.linspace(start[i], end[i], steps=nsamples[i]) 
                   for i in range(dim)])
    grid = torch.stack(torch.meshgrid(*dir_samples, indexing='ij'), dim=-1)
    return grid.reshape(-1, dim) if flatten else grid

def rotation_matrix(axis, theta):
    """
    This function returns a 3x3 rotation matrix of an angle theta around one of the main axes in 3D.
    
    Args:
    - axis: str, the axis to rotate around, 'x', 'y', or 'z'
    - theta: float, the angle in radians to rotate by
    
    Returns:
    - rot_mat: torch.Tensor of shape (3,3), the rotation matrix
    """
    assert axis in ['x', 'y', 'z'], "Axis must be 'x', 'y', or 'z'"
    
    if axis == 'x':
        rot_mat = torch.tensor([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
    elif axis == 'y':
        rot_mat = torch.tensor([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    else: # axis == 'z'
        rot_mat = torch.tensor([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    
    return rot_mat.float()

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

# only works for plane slices in 3D for now
def make_domain_slices(nsamples, start, end, dircodes):
    valid_codes = ['x', 'y', 'z', 'xy', 'xz', 'yz']
    code_map = {
        'x': [2, 0, 1],
        'y': [0, 2, 1],
        'z': [0, 1, 2]
    }
    value_map = {'x': 1, 'y': 2, 'z': 3}
    
    coords = make_grid_coords(nsamples, start, end, 2)
    slices = []
    for code in dircodes:
        if code not in valid_codes:
            raise ValueError(
                "Direction codes should be in [x, y, z, xy, xz, yz]")
        newdim = value_map[code[0]] * torch.ones((len(coords), 1))
        domain_slice = torch.cat([coords, newdim], dim=-1)
        domain_slice = domain_slice[:, code_map[code[0]]]
        if len(code) == 2:
            # todo, make angle a parameter
            rotation = rotation_matrix(code[1], np.pi/4)
            domain_slice = torch.matmul(rotation, 
                                        domain_slice.permute((1, 0))
                                        ).permute((1, 0))
        slices.append(domain_slice.reshape(nsamples, nsamples, 3))

    return slices

# def checker(res, scale=10):
#     tensors = tuple(3 * [torch.linspace(-1, 1, steps=res)])
#     grid = torch.stack(torch.meshgrid(*tensors, indexing='ij'), dim=-1)
#     grid = grid.reshape(-1, 3)

#     volume = torch.sin(scale * grid)
#     volume = grid[:, 0] * grid[:, 1] * grid[:, 2]
#     volume[volume < 0] = 0.0
#     volume[volume > 0] = 1.0
#     return volume.view(res, res, res).numpy()

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
    from PIL import Image

    # Define texture size and cube size
    # texture_size = 128
    # cube_size = 16

    # Combine the sine waves and threshold to create checkerboard pattern
    # board = ((sine_x + sine_y + sine_z) > 0).astype(np.uint8) * 255
    # print(board.shape)
    # board = (solid_texture(texture_size, 0.4) * 255).astype(np.uint8)
    # Image.fromarray(board[0, :, :]).save('x.png')
    # Image.fromarray(board[:, 0, :]).save('y.png')
    # Image.fromarray(board[:, :, 0]).save('z.png')
    # # Convert NumPy array to PIL image
    # image = Image.fromarray(checkerboard)

    # # Save the generated texture
    # image.save('checkerboard_sine_3d.png')
    slices = make_domain_slices(32, -1, 1, ['x', 'y', 'z', 'xy'])
    print(len(slices), slices[0].shape)

    