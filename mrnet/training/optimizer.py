import torch
import torch.nn.functional as F


class OptimizationHandler:
    def __init__(self, model, optimizer, loss_function, loss_weights) -> None:
        self.model = model
        # TODO: should it be constructed here?
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.loss_weights = loss_weights
       
    def _pre_process(self):
        self.optimizer.zero_grad()
        # not all model's parameters are updated by the optimizer
        self.model.zero_grad()

    def _post_process(self, loss_dict):
        loss = sum([loss_dict[key] * self.loss_weights[key] 
                    for key in loss_dict.keys()])

        loss.backward()
        self.optimizer.step()

        running_loss = {}
        for key, value in loss_dict.items():
            running_loss[key] = (running_loss.get(key, 0.0) 
                                    + value.item())
        return running_loss
            
    def optimize(self, batch, device):
        """This function should be overwritten for custom losses"""
        # why c0?
        X, gt_dict = batch['c0']
        out_dict = self.model(X['coords'].to(device))
        
        loss_dict = self.loss_function(out_dict, gt_dict, device=device)
        return loss_dict

    def __call__(self, batch, device):
        self._pre_process()
        loss_dict = self.optimize(batch, device)
        return self._post_process(loss_dict)
    
class MirrorOptimizationHandler(OptimizationHandler):
    def optimize(self, batch, device):
        X, gt_dict = batch['c0']
        out_dict = self.model(X['coords'].to(device))
        loss_dict = self.loss_function(out_dict, gt_dict, device=device)
        
        # TODO: revise it; looks like the logic is not good
        mirror_loss = 0.0
        offset = self.model.period / 2
        for k in range(self.model.in_features + 1):
            if k == 0:
                dirx = 2 - X['coords'][..., 0]
                diry = 0 + X['coords'][..., 1]
            elif k == 1:
                dirx = 0 + X['coords'][..., 0]
                diry = 2 - X['coords'][..., 1]
            else: 
                dirx = 2 - X['coords'][..., 0]
                diry = 2 - X['coords'][..., 1]
            mirror_x =  torch.stack([dirx, diry], dim=-1)
            out_mirror = self.model(mirror_x.to(device))
            mirror_loss += F.mse_loss(out_mirror['model_out'],
                                    out_dict['model_out'])
        loss_dict['mirror'] = mirror_loss / 2
        return loss_dict