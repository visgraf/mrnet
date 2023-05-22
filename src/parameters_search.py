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
from datasets.procedural import voronoi_texture, marble_texture, marble_color
from copy import deepcopy


BASE_DIR = Path('.').absolute()
VOXEL_PATH = BASE_DIR.joinpath('vox')
MODEL_PATH = BASE_DIR.joinpath('models')

def procedural_multiresolution(base_signal, max_stages, decimate, seed):
    mr_stack = [base_signal]
    domain_length = 2 #base_signal.domain[1] - base_signal.domain[0]
    base_dim = base_signal.shape[-1]
    cmap = lambda k: marble_color(torch.sin(2 * k * torch.pi))
    for i in range(1, max_stages):
        torch.manual_seed(seed)
        new_dim = (base_dim // 2**i)
        pixelsize = domain_length / new_dim

        procedure = marble_texture(pixelsize, cmap)
        dims = ((new_dim, new_dim, new_dim) if decimate 
                    else (base_dim, base_dim, base_dim))
        mr_stack.append(Procedural3DSignal(procedure,
                                           dims,
                                           base_signal.channels,
                                           base_signal.domain,
                                           base_signal.attributes,
                                           batch_size=base_signal.batch_size,
                                           color_space=base_signal.color_space))
    return mr_stack

def run_experiment(hyper, project_name, seed, mrmodel=None):
    torch.manual_seed(seed)
    
    dim = hyper['width']
    if hyper['filename'] == 'marble':
        cmap = lambda k: marble_color(torch.sin(2 * k * torch.pi))
        proc = marble_texture(2/dim, cmap)
    elif hyper['filename'] == 'voronoi':
        proc = voronoi_texture(16)
    
    base_signal = Procedural3DSignal(
        proc,
        (dim, dim, dim),
        channels=hyper['channels'],
        domain=hyper['domain'],
        batch_size=hyper['batch_size'],
        color_space=hyper['color_space']
    )
    train_dataset = procedural_multiresolution(base_signal, 
                                               hyper['max_stages'], 
                                               False,
                                               seed)
    test_dataset = procedural_multiresolution(base_signal, 
                                               hyper['max_stages'], 
                                               False,
                                               seed)

    filename = os.path.basename(hyper['filename'])
    wandblogger = WandBLogger3D(project_name,
                                f"{hyper['model']}{hyper['filter'][0].upper()}{filename[0:5]}",
                                hyper,
                                BASE_DIR)
    if mrmodel is None:
        mrmodel = MRFactory.from_dict(hyper)
    print("Model: ", type(mrmodel))
    mrtrainer = MRTrainer.init_from_dict(mrmodel, 
                                         train_dataset, test_dataset, wandblogger, hyper)
    mrtrainer.train(hyper['device'])
    path = 'C:\\Users\\hallpaz\\Workspace\\mrnet\\models\\siggraph'
    MRFactory.save(mrmodel, os.path.join(path, f'marble{mrmodel.n_stages()}.pth'))


if __name__ == '__main__':
    project_name = "solid-incremental"
    config_file = 'configs/siggraph_asia/config_solid_texture.yml'
    with open(config_file) as f:
        hyper = yaml.load(f, Loader=SafeLoader)
        if isinstance(hyper['batch_size'], str):
            hyper['batch_size'] = eval(hyper['batch_size'])
        if hyper['channels'] == 0:
            hyper['channels'] = hyper['out_features']
        print(hyper)

    hyper['width'] = 512
    hyper['height'] = 512
    hyper['depth'] = 512
    hyper['max_stages'] = 1
    hyper['omega_0'] = [64, 24, 32, 64, 96]
    hyper['hidden_features'] = [[2048, 1024, 512], [256, 512], [512, 512], [1024, 512], [1024, 512], [512, 1024]]
    hyper['color_space'] = 'YCbCr'
    hyper['max_epochs_per_stage'] = [14, 5, 5, 5, 5]
    run_experiment(hyper, project_name, 777)

    # for hidden_features in [
    #                          [[512, 256], [768, 384], [1024, 512], [1536, 768], [2048, 1024]],
    #                          [512, 768, 1024, 1280, 1600],
    #                          [512, 1024, 1536, 2048, 2048],
    #                          [512, [1024, 512], [1024, 512], [2048, 1024], [2048, 1024]] ]:
    #     exp = {'hidden_features': hidden_features}
    #     exp_hyper = deepcopy(hyper)
    #     exp_hyper.update(exp)
    #     run_experiment(exp_hyper, project_name, seed=777)
    
    # for hidden_features in [[2048, 1024, 1024, 512], [2048, 1024, 512, 256], [256, 512, 1024]]:
    #     exp = {}
    #     if isinstance(hidden_features, Sequence):
    #         exp['hidden_layers'] = len(hidden_features) - 1
    #     else:
    #         exp['hidden_layers'] = 1
    #     exp['hidden_features'] = [hidden_features]
    #     for omega_0 in [8, 16, 24, 32]:
    #         exp['omega_0'] = [omega_0]
    #         for res in [256, 512]:
    #             exp["width"] = res 
    #             exp["height"] = res
    #             exp["depth"] = res
    #             try:
    #                 exp_hyper = deepcopy(hyper)
    #                 exp_hyper.update(exp)
    #                 run_experiment(exp_hyper, 
    #                                project_name, seed=777)
    #             except Exception as e:
    #                 print(e, hidden_features, omega_0, res)
    #                 wandb.finish()
                    