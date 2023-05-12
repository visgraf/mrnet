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

    loss_dict['d0'] = F.mse_loss(pred, gt_pred) 
    # loss_dict['d1'] = F.l1_loss(pred_grad, gt_grad)
    # loss_dict['d1'] = F.mse_loss(pred_grad, gt_grad)
    loss_dict['d1'] = torch.mean((pred_grad - gt_grad).pow(2).sum(-1))
    return loss_dict

def get_loss_from_map(lossname:str):
    valid_losses = {
        'mse': mse_loss,
        'hermite': hermite_loss
    }
    return valid_losses[lossname]

# def old_mse_loss(batch, model, mrweights, device):
#     loss_dict = {}

#     pred_list = []
#     gt_list = []

#     output_dict, trainY = perform_inference(batch, model, mrweights, device, 'c0')

#     train_dict = {k: v.to(device) for k, v in trainY.items()}
#     pred_list.append(output_dict['model_out'])
#     gt_list.append(train_dict['d0'])
    
#     pred = torch.cat(pred_list, dim=1)
#     gt = torch.cat(gt_list, dim=1)

#     loss_dict['d0'] = F.mse_loss(pred, gt) 

#     return loss_dict

# def old_hermite_loss(batch, model, mrweights,device):
#     (trainX, trainY) = batch['c1']
#     output_dict = model(trainX['coords'].to(device), mrweights=mrweights)
#     train_dict = {k: v.to(device) for k, v in trainY.items()}
#     coords = output_dict['model_in']
    
#     pred = output_dict['model_out']
#     gt_grad = train_dict['d1']
    
#     pred_grad = gradient(pred, coords)
#     pred_grad_flat = pred_grad.view(1,-1,1)

#     loss_dict = mse_loss(batch, model, mrweights, device)
#     loss_dict['d1'] = F.mse_loss(pred_grad_flat, gt_grad)/2

#     return loss_dict

# def perform_inference(batch, model, mrweights,device, class_points):
#     (trainX, trainY) = batch[class_points]
#     output_dict = model(trainX['coords'].to(device), mrweights=mrweights)

#     return output_dict, trainY