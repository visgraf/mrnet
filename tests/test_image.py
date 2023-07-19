import os
from pathlib import Path
import torch
from mrnet.logs.wandblogger import WandBLogger2D
from mrnet.training.trainer import MRTrainer
from mrnet.datasets.signals import ImageSignal
from mrnet.networks.mrnet import MRFactory
from mrnet.datasets.pyramids import create_MR_structure
import yaml
from yaml.loader import SafeLoader
import os

os.environ["WANDB_NOTEBOOK_NAME"] = "train-wb.ipynb"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
BASE_DIR = Path('.').absolute()
IMAGE_PATH = BASE_DIR.joinpath('img')
MODEL_PATH = BASE_DIR.joinpath('models')
torch.manual_seed(777)

#-- hyperparameters in configs --#
config_file = 'configs/tests/image.yml'
with open(config_file) as f:
    hyper = yaml.load(f, Loader=SafeLoader)
    if isinstance(hyper['batch_size'], str):
        hyper['batch_size'] = eval(hyper['batch_size'])
    if hyper.get('channels', 0) == 0:
            hyper['channels'] = hyper['out_features']
    print(hyper)
imgpath = os.path.join(IMAGE_PATH, hyper['image_name'])
project_name = hyper.get('project_name', 'dev_sandbox')
maskpath = None
hyper['device']

base_signal = ImageSignal.init_fromfile(
                    imgpath,
                    domain=hyper['domain'],
                    channels=hyper['channels'],
                    sampling_scheme=hyper['sampling_scheme'],
                    width=hyper['width'], height=hyper['height'],
                    attributes=hyper['attributes'],
                    batch_size=hyper['batch_size'],
                    color_space=hyper['color_space'])

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

if hyper['width'] == 0:
    hyper['width'] = base_signal.shape[-1]
if hyper['height'] == 0:
    hyper['height'] = base_signal.shape[-1]

img_name = os.path.basename(hyper['image_name'])
mrmodel = MRFactory.from_dict(hyper)
print("Model: ", type(mrmodel))
wandblogger = WandBLogger2D(project_name,
                            f"{hyper['model']}{hyper['filter'][0].upper()}{img_name[0:5]}{hyper['color_space'][0]}",
                            hyper,
                            BASE_DIR)
mrtrainer = MRTrainer.init_from_dict(mrmodel, 
                                     train_dataset, 
                                     test_dataset, 
                                     wandblogger, 
                                     hyper)
mrtrainer.train(hyper['device'])