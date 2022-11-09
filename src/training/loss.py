import torch
import torch.nn.functional as F

def perform_inference(batch, model, mrweights,device, class_points):
    (trainX, trainY) = batch[class_points]
    output_dict = model(trainX['coords'].to(device),mrweights=mrweights)

    return output_dict, trainY

def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

def mse_loss(batch, model, mrweights,device):
    loss_dict = {}

    pred_list = []
    gt_list = []

    for key in batch.keys():
        output_dict, trainY = perform_inference(batch, model, mrweights,device, key)

        train_dict = {k: v.to(device) for k, v in trainY.items()}
        pred_list.append(output_dict['model_out'])
        gt_list.append(train_dict['d0'])
    
    pred = torch.cat(pred_list, dim=1)
    gt = torch.cat(gt_list, dim=1)

    loss_dict['d0'] = F.mse_loss(pred, gt) 

    return loss_dict

def hermite_loss(batch, model, mrweights,device):


    (trainX, trainY) = batch['c1']
    output_dict = model(trainX['coords'].to(device),mrweights=mrweights)
    train_dict = {k: v.to(device) for k, v in trainY.items()}
    coords = output_dict['model_in']
    
    pred = output_dict['model_out']
    gt_grad = train_dict['d1']
    
    pred_grad = gradient(pred, coords)
    pred_grad_flat = pred_grad.view(1,-1,1)

    loss_dict = mse_loss(batch, model, mrweights,device)
    loss_dict['d1'] = F.mse_loss(pred_grad_flat, gt_grad)

    return loss_dict

def get_loss_from_map(lossname:str):

    if lossname=='mse':
        return mse_loss

    if lossname=='hermite':
        return hermite_loss