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
