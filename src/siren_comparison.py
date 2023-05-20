import os
from pathlib import Path
import torch
import yaml
from yaml.loader import SafeLoader
from logs.wandblogger import WandBLogger2D
from training.trainer import MRTrainer
from datasets.signals import ImageSignal
from networks.mrnet import MRFactory


BASE_DIR = Path('.').absolute()
IMAGE_PATH = BASE_DIR.joinpath('img')
torch.manual_seed(777)

def load_config_file(config_path):
    with open(config_path) as f:
        hyper = yaml.load(f, Loader=SafeLoader)
        if isinstance(hyper['batch_size'], str):
            hyper['batch_size'] = eval(hyper['batch_size'])
        if hyper.get('channels', 0) == 0:
            hyper['channels'] = hyper['out_features']
        print(hyper)
    return hyper

if __name__ == '__main__':
    project_name = "siggraph_asia"
    #-- hyperparameters in configs --#
    config_file = 'configs/siggraph_asia/config_siggraph_siren.yml'
    hyper = load_config_file(config_file)
    imgpath = os.path.join(IMAGE_PATH, hyper['image_name'])
    
    base_signal = ImageSignal.init_fromfile(
                            imgpath,
                            domain=hyper['domain'],
                            channels=hyper['channels'],
                            sampling_scheme=hyper['sampling_scheme'],
                            width=hyper['width'], 
                            height=hyper['height'],
                            attributes=hyper['attributes'],
                            batch_size=hyper['batch_size'],
                            color_space=hyper['color_space'])
    # no multiresolution
    train_dataset = [base_signal]
    test_dataset = [base_signal]

    for period in [0, 2]:
        hyper['period'] = period
        mrmodel = MRFactory.from_dict(hyper)

        kind = 'Sir' if hyper['period'] == 0 else 'M'
        img_name = os.path.basename(hyper['image_name'])
        wandblogger = WandBLogger2D(project_name,
                                    f"{kind}{img_name[0:5]}{hyper['color_space'][0]}",
                                    hyper,
                                    BASE_DIR)
        mrtrainer = MRTrainer.init_from_dict(mrmodel, 
                                            train_dataset, 
                                            test_dataset, 
                                            wandblogger, 
                                            hyper)
        mrtrainer.train(hyper['device'])