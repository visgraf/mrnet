from typing import Sequence
import torch
import os
from pathlib import Path

import wandb
from logs.wandblogger import WandBLogger3D 
from training.trainer import MRTrainer
from datasets.signals import Procedural3DSignal
from networks.mrnet import MRFactory
import yaml
from yaml.loader import SafeLoader
from datasets.procedural import voronoi_texture, marble_texture
from copy import deepcopy


os.environ["WANDB_NOTEBOOK_NAME"] = "train3d.ipynb"
BASE_DIR = Path('.').absolute().parents[0]
VOXEL_PATH = BASE_DIR.joinpath('vox')
MODEL_PATH = BASE_DIR.joinpath('models')

def run_experiment(hyper, project_name, seed):
    torch.manual_seed(seed)
    
    dim = hyper['width']
    if hyper['filename'] == 'marble':
        proc = marble_texture(2/dim)
    elif hyper['filename'] == 'voronoi':
        proc = voronoi_texture(16)
    
    base_signal = Procedural3DSignal(
        proc,
        (dim, dim, dim),
        channels=hyper['channels'],
        domain=hyper['domain'],
        batch_size=hyper['batch_size']
    )
    train_dataset = [base_signal]
    test_dataset = [base_signal]

    filename = os.path.basename(hyper['filename'])
    wandblogger = WandBLogger3D(project_name,
                                f"{hyper['model']}{hyper['filter'][0].upper()}{filename[0:5]}",
                                hyper,
                                BASE_DIR)
    mrmodel = MRFactory.from_dict(hyper)
    print("Model: ", type(mrmodel))
    mrtrainer = MRTrainer.init_from_dict(mrmodel, 
                                         train_dataset, test_dataset, wandblogger, hyper)
    mrtrainer.train(hyper['device'])


if __name__ == '__main__':
    project_name = "params-search"
    config_file = 'configs/config_3d_m_net.yml'
    with open(config_file) as f:
        hyper = yaml.load(f, Loader=SafeLoader)
        if isinstance(hyper['batch_size'], str):
            hyper['batch_size'] = eval(hyper['batch_size'])
        print(hyper)
    
    
    for hidden_features in [[2048, 1024, 1024, 512], [2048, 1024, 512, 256], [256, 512, 1024]]:
        exp = {}
        if isinstance(hidden_features, Sequence):
            exp['hidden_layers'] = len(hidden_features) - 1
        else:
            exp['hidden_layers'] = 1
        exp['hidden_features'] = [hidden_features]
        for omega_0 in [8, 16, 24, 32]:
            exp['omega_0'] = [omega_0]
            for res in [256, 512]:
                exp["width"] = res 
                exp["height"] = res
                exp["depth"] = res
                try:
                    exp_hyper = deepcopy(hyper)
                    exp_hyper.update(exp)
                    run_experiment(exp_hyper, 
                                   project_name, seed=777)
                except Exception as e:
                    print(e, hidden_features, omega_0, res)
                    wandb.finish()
                    