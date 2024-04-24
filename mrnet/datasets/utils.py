import torch 
import numpy as np
from typing import Sequence, Union
from torch.utils.data import BatchSampler
from PIL import Image


RESIZING_FILTERS = {
    'nearest': Image.Resampling.NEAREST,
    'linear': Image.Resampling.BILINEAR,
    'cubic': Image.Resampling.BICUBIC,
}


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
    grid = torch.stack(torch.meshgrid(*dir_samples, indexing='xy'), dim=-1)
    return grid.reshape(-1, dim) if flatten else grid

# only works for plane slices in 3D for now
def make_domain_slices(nsamples, start, end, slice_views, slice_idx={}):
    valid_codes = ['x', 'y', 'z', 'xy', 'xz', 'yz']
    code_map = {
        'x': [2, 0, 1],
        'y': [0, 2, 1],
        'z': [0, 1, 2]
    }
    if not slice_idx:
        slice_idx = {'x': 1, 'y': 2, 'z': 3}
    
    coords = make_grid_coords(nsamples, start, end, 2)
    slices = []
    for code in slice_views:
        if code not in valid_codes:
            raise ValueError(
                "Direction codes should be in [x, y, z, xy, xz, yz]")
        try:
            idx = slice_idx[code]
            value = torch.linspace(start, end, nsamples)[idx]
        except KeyError:
            value = 0.0
        newdim = value * torch.ones((len(coords), 1))
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

def output_on_batched_dataset(model, dataset, device):
    model_out = []
    with torch.no_grad():
        for batch in dataset:
            input, _ = batch['c0']
            output_dict = model(input['coords'].to(device))
            model_out.append(output_dict['model_out'])
    return torch.concat(model_out)

def output_on_batched_grid(model, grid, batch_size, device):
    output = []
    for batch in BatchSampler(grid, batch_size, drop_last=False):
        batch = torch.stack(batch).to(device)
        output.append(model(batch)['model_out'])
    return torch.concat(output)

def output_on_batched_points(model, points, batch_size, device):
    output = []
    for batch in BatchSampler(points, batch_size, drop_last=False):
        batch = torch.stack(batch).to(device)
        output.append(model(batch)['model_out'])
    return torch.concat(output)

def ycbcr_to_rgb(image: torch.Tensor, channel_dim=-1) -> torch.Tensor:
    r"""Convert an YCbCr image to RGB.

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: YCbCr Image to be converted to RGB with shape :math:`(H, W, 3)`.

    Returns:
        RGB version of the image with shape :math:`(H, W, 3)`.
    based on: https://kornia.readthedocs.io/en/latest/_modules/kornia/color/ycbcr.html
    """
    if channel_dim == -1:
        y = image[..., 0]
        cb = image[..., 1]
        cr = image[..., 2]
    elif channel_dim == 0:
        y = image[0, ...]
        cb = image[1, ...]
        cr = image[2, ...]
    else:
        raise ValueError(f"Invalid channel_dim: {channel_dim}")

    delta: float = 0.5
    cb_shifted = cb - delta
    cr_shifted = cr - delta

    r = y + 1.403 * cr_shifted
    g = y - 0.714 * cr_shifted - 0.344 * cb_shifted
    b = y + 1.773 * cb_shifted
    return torch.stack([r, g, b], channel_dim)

def rgb_to_grayscale(image, channel_dim=-1):
    if channel_dim == -1:
        r = image[..., 0:1]
        g = image[..., 1:2]
        b = image[..., 2:3]
    elif channel_dim == 0:
        r = image[0:1, ...]
        g = image[1:2, ...]
        b = image[2:3, ...]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def rgb_to_ycbcr(image: torch.Tensor, channel_dim=-1) -> torch.Tensor:
    r"""Convert an RGB image to YCbCr.

    Args:
        image: RGB Image to be converted to YCbCr with shape :math:`(*, 3, H, W)`.

    Returns:
        YCbCr version of the image with shape :math:`(*, 3, H, W)`.

    """
    if channel_dim == -1:
        r = image[..., 0]
        g = image[..., 1]
        b = image[..., 2]
    elif channel_dim == 0:
        r = image[0, ...]
        g = image[1, ...]
        b = image[2, ...]

    delta: float = 0.5
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = (b - y) * 0.564 + delta
    cr = (r - y) * 0.713 + delta
    return torch.stack([y, cb, cr], channel_dim)

COLOR_MAPPING = {
    'RGB': lambda x: x,
    'L': rgb_to_grayscale,
    'YCbCr': rgb_to_ycbcr
}

INVERSE_COLOR_MAPPING = {
    'RGB': lambda x: x,
    'L': lambda x: x,
    'YCbCr': lambda x: ycbcr_to_rgb(x)
}

if __name__ == '__main__':
    from PIL import Image

    slices = make_domain_slices(32, -1, 1, ['x', 'y', 'z', 'xy'])
    print(len(slices), slices[0].shape)

    