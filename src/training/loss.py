import torch
import torch.nn.functional as F


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

def mse_loss(output_dict, train_dict):
    pred = output_dict['model_out']
    gt = train_dict['d0']

    loss_dict = {}
    loss_dict['d0'] = F.mse_loss(pred, gt)

    return loss_dict



def hermite_loss(output_dict, train_dict):
    coords = output_dict['model_in']
    
    pred = output_dict['model_out']
    gt_grad = train_dict['d1']
    
    pred_grad = gradient(pred, coords)
    pred_grad_flat = pred_grad.view(1,-1,1)

    loss_dict = {}
    loss_dict['d1'] = F.mse_loss(pred_grad_flat, gt_grad)

    return loss_dict

def get_loss_from_map(lossname:str):

    if lossname=='mse':
        return mse_loss

    if lossname=='hermite':
        return hermite_loss