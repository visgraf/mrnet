import os
from pathlib import Path
from logs.wandblogger import WandBLogger1D
from training.trainer import MRTrainer
from datasets.signals import Signal1D
from networks.mrnet import MRFactory
import yaml
from yaml.loader import SafeLoader
import torch


torch.manual_seed(777)
BASE_DIR = Path('.').absolute().parents[0]
DATA_PATH = BASE_DIR.joinpath('img')
MODEL_PATH = BASE_DIR.joinpath('models')

def summed_frequencies(x, freqs):
    res = torch.zeros_like(x)
    for k in freqs:
        res += torch.sin(k * 2*torch.pi * x)
    return res

def allmul_frequencies(x, freqs):
    res = torch.ones_like(x)
    for k in freqs:
        res *= torch.sin(k * 2*torch.pi * x)
    return res

def onemul_frequencies(x, freqs):
    res = summed_frequencies(x, freqs[:-1])
    m = freqs[-1]
    return torch.sin(m * 2*torch.pi * x) * res

def composition(x, freqs):
    # res = summed_frequencies(x, freqs[:-1])
    res = summed_frequencies(x, freqs)
    # m = freqs[-1]
    return torch.sin(2*torch.pi * res)

def load_config_file(config_path):
    with open(config_path) as f:
        hyper = yaml.load(f, Loader=SafeLoader)
        if isinstance(hyper['batch_size'], str):
            hyper['batch_size'] = eval(hyper['batch_size'])
        if hyper.get('channels', 0) == 0:
            hyper['channels'] = hyper['out_features']
        print(hyper)
    return hyper

def sin_mul_frequencies(x, freqs):
    res = allmul_frequencies(x, freqs)
    return torch.sin(x) * res

func_map = {'S': summed_frequencies, 
            'A': allmul_frequencies, 
            'O': onemul_frequencies,
            'M': sin_mul_frequencies, 
            'C': composition}

if __name__ == '__main__':
    project_name = "siggraph_asia"
    #-- hyperparameters in configs --#
    config_file = 'configs/siggraph_asia/config_frequency_analysis.yml'
    hyper = load_config_file(config_file)
    
    frequencies = [2, 7, 4]
    x = torch.linspace(-1, 1, hyper['width'])
    for code in func_map.keys():
        synthetic = func_map[code](x, frequencies)
        base_signal = Signal1D(synthetic.view(1, -1),
                                domain=hyper['domain'],
                                sampling_scheme=hyper['sampling_scheme'],
                                attributes=hyper['attributes'],
                                batch_size=hyper['batch_size'])
        # no multiresolution
        train_dataloader = [base_signal]  
        test_dataloader = [base_signal]

        mrmodel = MRFactory.from_dict(hyper)
        print("Model: ", type(mrmodel))

        img_name = os.path.basename(hyper['image_name'])
        wandblogger = WandBLogger1D(project_name,
                                    f"{hyper['model']}{code}{img_name[0:5]}",
                                    hyper,
                                    BASE_DIR)
        mrtrainer = MRTrainer.init_from_dict(mrmodel, train_dataloader, test_dataloader, wandblogger, hyper)
        mrtrainer.train(hyper['device'])