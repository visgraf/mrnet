import os
from pathlib import Path
import torch
from mrnet.training.trainer import MRTrainer
from mrnet.datasets.signals import ImageSignal
from mrnet.networks.mrnet import MRFactory
from mrnet.datasets.pyramids import create_MR_structure
from mrnet.logs.listener import TrainingListener
import yaml
from yaml.loader import SafeLoader
import os

os.environ["WANDB_NOTEBOOK_NAME"] = "train-wb.ipynb"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
BASE_DIR = Path('.').absolute()
IMAGE_PATH = BASE_DIR.joinpath('img')
MODEL_PATH = BASE_DIR.joinpath('models')
torch.manual_seed(777)

def load_hyperparameters(config_file):
    #-- hyperparameters in configs --#
    with open(config_file) as f:
        hyper = yaml.load(f, Loader=SafeLoader)
        if isinstance(hyper['batch_size'], str):
            hyper['batch_size'] = eval(hyper['batch_size'])
        if hyper.get('channels', 0) == 0:
                hyper['channels'] = hyper['out_features']
        print(hyper)
    maskpath = None
    return hyper

def load_datasets(hyper):
    imgpath = os.path.join(IMAGE_PATH, hyper['image_name'])
    base_signal = ImageSignal.init_fromfile(
                        imgpath,
                        domain=hyper['domain'],
                        channels=hyper['channels'],
                        sampling_scheme=hyper['sampling_scheme'],
                        width=hyper['width'], height=hyper['height'],
                        attributes=hyper['attributes'],
                        batch_size=hyper['batch_size'],
                        color_space=hyper['color_space'])
    
    if hyper['width'] == 0:
        hyper['width'] = base_signal.shape[-1]
    if hyper['height'] == 0:
        hyper['height'] = base_signal.shape[-1]

    train_dataset = create_MR_structure(base_signal, 
                                        hyper['max_stages'], 
                                        hyper['filter'], 
                                        hyper['decimation'],
                                        hyper['pmode'])
    test_dataset = create_MR_structure(base_signal, 
                                        hyper['max_stages'], 
                                        hyper['filter'], 
                                        False,
                                        hyper['pmode'])
    return train_dataset, test_dataset

def train_and_log(hyper, train_dataset, test_dataset):
    project_name = hyper.get('project_name', 'tests')
    img_name = os.path.basename(hyper['image_name'])
    mrmodel = MRFactory.from_dict(hyper)
    print("Model: ", type(mrmodel))
    wandblogger = TrainingListener(project_name,
                                f"{hyper['model']}{hyper['filter'][0].upper()}{img_name[0:5]}{hyper['color_space'][0]}",
                                hyper,
                                BASE_DIR)
    mrtrainer = MRTrainer.init_from_dict(mrmodel, 
                                        train_dataset, 
                                        test_dataset, 
                                        wandblogger, 
                                        hyper)
    mrtrainer.train(hyper['device'])


def test(change_params, device, config_file='configs/tests/image.yml'):
    hyper = load_hyperparameters(config_file)
    hyper['device'] = device
    change_params(hyper)
    train_dataset, test_dataset = load_datasets(hyper)
    train_and_log(hyper, train_dataset, test_dataset)

def grayscale_color_space(hyper):
    hyper['color_space'] = 'L'
    hyper['out_features'] = 1
    hyper['channels'] = 1

def rgb_color_space(hyper):
    hyper['color_space'] = 'RGB'
    hyper['out_features'] = 3
    hyper['channels'] = 3

def ycbcr_color_space(hyper):
    hyper['color_space'] = 'YCbCr'
    hyper['out_features'] = 3
    hyper['channels'] = 3

def multistage_architecture(hyper):
    hyper['hidden_features'] = [[64, 48], [128, 96], [256, 128]]
    hyper['omega_0'] = [8, 16, 32]
    hyper['hidden_omega_0'] = [30, 30, 30]
    hyper['max_stages'] = 3

def lnet_architecture(hyper):
    hyper['model'] = 'L'
    multistage_architecture(hyper)
    
def mnet_architecture(hyper):
    hyper['model'] = 'M'
    multistage_architecture(hyper)

def not_periodic(hyper):
    mnet_architecture(hyper)
    hyper['period'] = -1
    hyper['superposition_w0'] = True

if __name__ == '__main__':
     for device in ['cpu', 'cuda']:
        test(grayscale_color_space, device)
        test(rgb_color_space, device)
        test(ycbcr_color_space, device)
        test(lnet_architecture, device)
        test(mnet_architecture, device)
        test(not_periodic, device)
        
        print('*' * 14, f"\nAll tests on {device} succeed!\n", '*' * 14, '\n')