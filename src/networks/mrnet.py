import torch
import numpy as np
import warnings

from .siren import SineLayer
from torch import nn
from torch.nn.parameter import Parameter
from typing import Iterator, Sequence
from copy import deepcopy


class MRModule(nn.Module):
    """
    Build upon SIREN code
    """
    def __init__(self, in_features: int, 
                    hidden_features: int, 
                    hidden_layers: int, 
                    out_features: int, 
                    first_omega_0: int, 
                    hidden_omega_0=1, 
                    bias=False, 
                    prevknowledge=0):
        super().__init__()
        
        self.first_layer = SineLayer(in_features, hidden_features, bias=bias,
                                  is_first=True, omega_0=first_omega_0)

        middle = []
        middle.append(
            SineLayer(prevknowledge + hidden_features, hidden_features,
                                is_first=False, omega_0=hidden_omega_0)
        )

        for i in range(hidden_layers - 1):
            middle.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.middle_layers = nn.Sequential(*middle)
        
        self.final_linear = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            self.final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
            
   
    # Check if internal layers initialization is needed/correct
    def reset_weights(self):
        def reset_sinelayer(m):
            if isinstance(m, SineLayer):
                m.init_weights()
        self.apply(reset_sinelayer)

    @property
    def in_features(self):
        return self.first_layer.in_features

    @property
    def out_features(self):
        return self.final_linear.out_features

    @property
    def hidden_features(self):
        return self.first_layer.out_features

    @property
    def hidden_layers(self):
        return len(self.middle_layers)

    @property
    def omega_0(self):
        return self.first_layer.omega_0
    
    @property
    def omega_G(self):
        return self.middle_layers[0].omega_0

    def forward(self, coords, prevbasis=None):
        proj = self.first_layer(coords)
        basis = (self.middle_layers(proj) if prevbasis is None 
                else self.middle_layers(torch.cat([proj, prevbasis], dim=-1)) )
        out = self.final_linear(basis)
        return out, basis


class MRNet(nn.Module):
    """
    Build upon SIREN code
    """
    def __init__(self, in_features, 
                    hidden_features, 
                    hidden_layers, 
                    out_features,
                    first_omega_0, 
                    hidden_omega_0=1, 
                    bias=False, 
                    superposition_w0=True):
        super().__init__()

        self.superposition_w0 = superposition_w0
        self.in_features = in_features
        self.out_features = out_features

        first_module = MRModule(in_features, 
                                hidden_features, 
                                hidden_layers, 
                                out_features,
                                first_omega_0, 
                                hidden_omega_0,
                                bias=bias)

        self.stages = nn.ModuleList([first_module])

    def init_module_weights(self, mrmodule: MRModule):
        w0 = mrmodule.first_layer.omega_0
        prev_w0 = self.top_stage.first_layer.omega_0
        
        layer_shape = mrmodule.first_layer.linear.weight.shape
        hidden_feat, in_feat = layer_shape[0], layer_shape[1]          

        c = prev_w0/w0
        p = torch.zeros(hidden_feat, in_feat).uniform_(-1, 1)

        # transform the interval (0, 1] --> ( c, 1]
        # and                    [-1,0] --> [-1,-c]
        ca = (1-c)*p + c
        cb = (1-c)*p - c
        p = torch.where(p > 0, ca, cb)

        with torch.no_grad():
            mrmodule.first_layer.linear.weight.copy_(p)

  
    def _add_stage(self, first_omega_0, hidden_features, 
                        hidden_layers, hidden_omega_0,
                        prevknowledge):
       
        newstage = MRModule(self.in_features, 
                            hidden_features, 
                            hidden_layers, 
                            self.out_features,
                            first_omega_0, 
                            hidden_omega_0,
                            prevknowledge=prevknowledge
                            ).to(self.current_device())
        if not self.superposition_w0:
            self.init_module_weights(newstage)
        self.stages.append(newstage)

    def add_stage(self, first_omega_0, hidden_features, 
                        hidden_layers, hidden_omega_0=1):
        raise NotImplementedError

    def n_stages(self):
        return len(self.stages)

    @property
    def top_stage(self)-> MRModule:
        return self.stages[-1]

    def _aggregate_resolutions(self, mroutputs, mrweights):
        device = self.current_device()
        if mrweights is None:
            mrweights = torch.ones(self.n_stages(), device=device)
        # Different weights per sample
        if len(mrweights.shape) == len(mroutputs[0].shape):
            concatenated = torch.concat(mroutputs, 1)
            weighted = torch.mul(concatenated, mrweights)
            return torch.sum(weighted, 1).unsqueeze(-1)
        # Same weights for all samples
        aggr_layer = nn.Linear(self.n_stages(), self.out_features, 
                                bias=False, device=device)
        for i in range(self.out_features):
            with torch.no_grad():
                aggr_layer.weight[i] = mrweights
       
        aggregated = aggr_layer(torch.stack(mroutputs, dim=-1))
        return aggregated.squeeze(-1)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        if recurse:
            return super().parameters(recurse)
        
        return self.top_stage.parameters()

    def current_device(self) -> str:
        return next(self.parameters()).device

    def class_code(self):
        raise NotImplementedError


class MNet(MRNet):
    def init_from_dict(hyper):
        omega0, hidden_omega0 = hyper['omega_0'], hyper['hidden_omega_0']
        return MNet(
            hyper.get('in_features', 1),
            hyper['hidden_features'],
            hyper['hidden_layers'],
            hyper.get('out_features', 1),
            omega0[0] if isinstance(omega0, Sequence) else omega0,
            hidden_omega0[0] if isinstance(hidden_omega0, Sequence) else hidden_omega0,
            bias=hyper.get('bias', False),
        )

    def add_stage(self, first_omega_0, hidden_features, hidden_layers, hidden_omega_0):
        prev = self.top_stage.hidden_features
        return self._add_stage(first_omega_0, hidden_features, hidden_layers, hidden_omega_0, prev)
    
    def forward(self, coords, mrweights=None):
        # allows to take derivative w.r.t. input
        coords = coords.clone().detach().requires_grad_(True) 
        
        mroutputs = []
        basis = None
        for mrstage in self.stages:
            out, basis = mrstage(coords, basis)
            mroutputs.append(out)

        if self.training and mrweights is not None:
            warnings.warn(
                "Using custom weights during training is discouraged. Make sure you know what you're doing"
            )
        
        y = self._aggregate_resolutions(mroutputs, mrweights)
        return {"model_in": coords, "model_out": y}
    
    def class_code(self):
        return 'M'

class LNet(MRNet):

    def init_from_dict(hyper):
        omega0, hidden_omega0 = hyper['omega_0'], hyper['hidden_omega_0']
        return  LNet(
            hyper.get('in_features', 1),
            hyper['hidden_features'],
            hyper['hidden_layers'],
            hyper.get('out_features', 1),
            omega0[0] if isinstance(omega0, Sequence) else omega0,
            hidden_omega0[0] if isinstance(hidden_omega0, Sequence) else hidden_omega0,
            bias=hyper.get('bias', False),
            superposition_w0=hyper.get('superposition_w0', True)
        )

    def add_stage(self, first_omega_0, hidden_features, hidden_layers, hidden_omega_0):
        return self._add_stage(first_omega_0, hidden_features, hidden_layers, hidden_omega_0, 0)

    def forward(self, coords, mrweights=None):
        # allows to take derivative w.r.t. input
        coords = coords.clone().detach().requires_grad_(True) 
        mroutputs = []
        for stage in self.stages:
            out, _ = stage(coords)
            mroutputs.append(out)
        
        # we could use another layer for a weighted sum
        y = self._aggregate_resolutions(mroutputs, mrweights)
        return {"model_in": coords, "model_out": y}

    def class_code(self):
        return 'L'

class MRFactory:
    def from_dict(hyper):
        omega0, hidden_omega0 = hyper['omega_0'], hyper['hidden_omega_0']
        if hyper['model'] == 'M':
            MRClass = MNet
        elif hyper['model'] == 'L': 
            MRClass = LNet
        elif hyper['model'] == 'M1':
            MRClass = M1Net
        else:
            raise ValueError("model should be in ['M','L','M1']")

        hfeat, hlayers = hyper['hidden_features'], hyper['hidden_layers']
        return  MRClass(
            hyper.get('in_features', 1),
            hfeat[0] if isinstance(hfeat, Sequence) else hfeat,
            hlayers[0] if isinstance(hlayers, Sequence) else hlayers,
            hyper.get('out_features', 1),
            omega0[0] if isinstance(omega0, Sequence) else omega0,
            hidden_omega0[0] if isinstance(hidden_omega0, Sequence) else hidden_omega0,
            bias=hyper.get('bias', False),
            superposition_w0=hyper.get('superposition_w0', True)
        )

    def module_from_dict(hyper, idx=None):
        if idx == 0:
            prevknowledge = False
        else:
            prevknowledge = True if hyper['model'] in ['M', 'M1'] else False
        return MRModule(hyper['in_features'],
                        hyper['hidden_features'],
                        hyper['hidden_layers'],
                        hyper['out_features'],
                        hyper['omega_0'],
                        hyper['hidden_omega_0'],
                        hyper.get('bias', False),
                        prevknowledge
        )

    def save(model:MRNet, path:str):
        firstmodule = model.stages[0]
        
        omega_0 = [mod.omega_0 for mod in model.stages]
        hidden_omega_0 = [mod.omega_G for mod in model.stages]
        hidden_layers = [mod.hidden_layers for mod in model.stages]
        hidden_features = [mod.hidden_features for mod in model.stages]
        mdict = {
                'omega_0': omega_0,
                'hidden_omega_0': hidden_omega_0,
                'model': model.class_code(),
                'stages': model.n_stages(),
                'in_features': firstmodule.in_features,
                'out_features': firstmodule.out_features,
                'hidden_layers': hidden_layers,
                'hidden_features': hidden_features,
            }
        for stg in range(model.n_stages()):
            mdict[f'module{stg}_state_dict'] = model.stages[stg].state_dict()
        torch.save(mdict, path)

    def load_state_dict(filepath):
        checkpoint = torch.load(filepath)
        singledict = deepcopy(checkpoint)
        module_keys = ['omega_0', 'hidden_omega_0', 'hidden_features', 'hidden_layers']
        updict = {k: checkpoint[k][0] for k in module_keys}
        singledict.update(updict)
        model = MRFactory.from_dict(singledict)
        model_stages = []
        for stage in range(checkpoint['stages']):
            updict = {k: checkpoint[k][stage] for k in module_keys}
            singledict.update(updict)
            mrmodule = MRFactory.module_from_dict(singledict, stage)
            mrmodule.load_state_dict(
                checkpoint[f'module{stage}_state_dict'])
            model_stages.append(mrmodule)     
        
        model.stages = nn.ModuleList(model_stages)
        model.eval()
        return model
        