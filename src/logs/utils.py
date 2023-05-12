import torch
from datasets.sampler import make_grid_coords
from torch.utils.data import BatchSampler

def output_per_batch(model, dataset, device):
    model_out = []
    with torch.no_grad():
        for batch in dataset:
            input, _ = batch['c0']
            output_dict = model(input['coords'].to(device))
            model_out.append(torch.clamp(output_dict['model_out'], 0.0, 1.0))
    return torch.concat(model_out)

def output_on_batched_domain(model, nsamples, domain, dim, batch_size, device):
    grid = make_grid_coords(nsamples, *domain, dim)
    output = []
    for batch in BatchSampler(grid, batch_size, drop_last=False):
        batch = torch.stack(batch)
        with torch.no_grad():
            output.append(model(batch.to(device))['model_out'])
    return torch.concat(output)

def ycbcr_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an YCbCr image to RGB.

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: YCbCr Image to be converted to RGB with shape :math:`(H, W, 3)`.

    Returns:
        RGB version of the image with shape :math:`(H, W, 3)`.
    based on: https://kornia.readthedocs.io/en/latest/_modules/kornia/color/ycbcr.html
    """

    y = image[..., 0]
    cb = image[..., 1]
    cr = image[..., 2]

    delta: float = 0.5
    cb_shifted = cb - delta
    cr_shifted = cr - delta

    r = y + 1.403 * cr_shifted
    g = y - 0.714 * cr_shifted - 0.344 * cb_shifted
    b = y + 1.773 * cb_shifted
    return torch.stack([r, g, b], -1)
