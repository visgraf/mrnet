import torch
import torch.nn.functional as F
from math import comb

def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

def get_device(tensor: torch.Tensor):
    return f"cuda:{tensor.get_device()}" if tensor.get_device() >= 0 else "cpu"

def make_d0_loss(loss_func):
    def loss(output_dict, gt_dict, **kwargs):
        pred: torch.Tensor = output_dict['model_out']
        device = kwargs.get('device', get_device(pred))
        pred = pred.to(device)
        gt = gt_dict['d0'].to(device)

        loss_dict = {'d0': loss_func(pred, gt)}
        return loss_dict
    
    return loss

mse_loss = make_d0_loss(F.mse_loss)
l1_loss = make_d0_loss(F.l1_loss)

def hermite_loss(output_dict, train_dict, **kwargs):
    pred: torch.Tensor = output_dict['model_out']
    device = kwargs.get('device', get_device(pred)) 
    pred = pred.to(device)
    coords = output_dict['model_in'].to(device)
    
    gt_pred = train_dict['d0'].to(device)
    gt_grad = train_dict['d1'].to(device)
    pred_grad = gradient(pred, coords)

    loss_dict = {
        'd0': F.mse_loss(pred, gt_pred),
        'd1': F.mse_loss(pred_grad, gt_grad)
    }
    return loss_dict

def num_of_directions(dim):
    dirs = 0
    for i in range(dim):
        dirs += comb(dim, i + 1)
    return dirs

def get_loss_from_map(lossname:str):
    valid_losses = {
        'l1': l1_loss,
        'mse': mse_loss,
        'hermite': hermite_loss,
        # 'reflect': reflect_loss
    }
    return valid_losses[lossname]