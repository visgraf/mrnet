import torch
import torch.nn.functional as F


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

def filter_active_values(model_val, train_dict, d_code):
    mask = train_dict[f'{d_code}_mask']
    pred = model_val[mask]
    gt = train_dict[d_code][mask]
    return pred, gt

    

def hermite_MSE(output_dict, train_dict, orders):
    model_out, coords = output_dict['model_out'], output_dict['model_in']
    loss_dict = {}
    if 'd0' in orders:
        pred, gt = filter_active_values(model_out, train_dict, 'd0')
        loss_dict['d0'] = F.mse_loss(pred, gt)

    if 'd1' in orders:
        model_d1 = gradient(model_out, coords)
        pred, gt = filter_active_values(model_d1.view(1,-1,1), train_dict, 'd1')
        loss_dict['d1'] = F.mse_loss(pred, gt)

    return loss_dict

def get_loss_from_map(lossname:str):
    def loss_func(output_dict, train_dict):
        return hermite_MSE(output_dict, train_dict, lossname)
    return loss_func