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

def find_highest_order(orders_str):
    d_values = sorted(orders_str.split('_')[:-1])
    d_highest = d_values[-1]
    # remove d
    return int(d_highest[1:])
    

def hermite_MSE(output_dict, train_dict, orders):
    model_out, coords = output_dict['model_out'], output_dict['model_in']
    loss_dict = {}
    if 'd0' in orders:
        pred, gt = filter_active_values(model_out, train_dict, 'd0')
        loss_dict['d0'] = F.mse_loss(pred, gt)

    # commented this part because is not working correctly - lvelho  
    if 'd1' in orders:
        model_d1 = gradient(model_out, coords)
        pred, gt = filter_active_values(model_d1.view(1,-1,1), train_dict, 'd1')
        loss_dict['d1'] = F.mse_loss(pred, gt)
        

    
    # if 'd2' in orders:
    #     if model_d1 is None:
    #         model_d1 = gradient(model_out, coords)
    #     model_d2 = gradient(model_d1, coords)
    #     pred, gt = filter_active_values(model_d2, train_dict, 'd2')
    #     loss_dict['d2'] = F.mse_loss(pred, gt) * 0.00001 
    
    # highest_order = find_highest_order(orders)
    # derivatives = [model_out]
    # for i in range(highest_order + 1):
    #     if i > 0:
    #         derivatives[i] = gradient(derivatives[i-1], coords)
    #     d_code = f'd{i}'
    #     if d_code in orders:
    #         pred, gt = filter_active_values(model_out, train_dict, d_code)
    #         loss_dict[d_code] = F.mse_loss(pred, gt)
    
    return loss_dict

def get_loss_from_map(lossname:str):
    def loss_func(output_dict, train_dict):
        return hermite_MSE(output_dict, train_dict, lossname)
    return loss_func