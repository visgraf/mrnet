import torch
import torch.nn.functional as F


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

def mse_loss(output_dict, gt_dict, device):
    loss_dict = {}
    pred = output_dict['model_out'].to(device)
    gt = gt_dict['d0'].to(device)

    loss_dict['d0'] = F.mse_loss(pred, gt) 
    return loss_dict

def hermite_loss(output_dict, train_dict, device):
    loss_dict = {}
    coords = output_dict['model_in'].to(device)
    pred = output_dict['model_out'].to(device)
    gt_pred = train_dict['d0'].to(device)
    gt_grad = train_dict['d1'].to(device)
    
    pred_grad = gradient(pred, coords)

    # GAMBIARRA
    #mask = (torch.rand_like(pred) < 0.2).int()
    # mask = torch.ones_like(pred)
    loss_dict['d0'] = F.mse_loss(pred, gt_pred) 
    # loss_dict['d0'] = F.mse_loss(pred * mask, gt_pred * mask) 
    loss_dict['d1'] = F.mse_loss(pred_grad, gt_grad)
    # loss_dict['d1'] = F.mse_loss(pred_grad * (1-mask), gt_grad * (1-mask))
    return loss_dict

def get_loss_from_map(lossname:str):
    valid_losses = {
        'mse': mse_loss,
        'hermite': hermite_loss
    }
    return valid_losses[lossname]