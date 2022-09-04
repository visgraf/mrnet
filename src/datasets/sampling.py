import torch

def make2Dcoords(width, height, start=-1, end=1):
    lx = torch.linspace(-1, 1, steps=width)
    ly = torch.linspace(-1, 1, steps=height)
    xs, ys = torch.meshgrid(lx,ly)
    return torch.stack([xs, ys], -1).view(-1, 2)

