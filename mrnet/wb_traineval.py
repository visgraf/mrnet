import os
from pathlib import Path

from logs.wandblogger import WandBLogger2D
from training.trainer import MRTrainer
from datasets.imagesignal import ImageSignal
from networks.mrnet import MRFactory
from datasets.pyramids import create_MR_structure
import yaml
from yaml.loader import SafeLoader
import os


os.environ["WANDB_NOTEBOOK_NAME"] = "wb_traineval.py"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
PYTORCH_ENABLE_MPS_FALLBACK = 1
BASE_DIR = Path('.').absolute()
IMAGE_PATH = BASE_DIR.joinpath('img')


def create_dataloaders(hyper):
    base_signal = ImageSignal.init_fromfile(
                        os.path.join(IMAGE_PATH, hyper['image_name']),
                        batch_samples_perc=hyper['batch_samples_perc'],
                        sampling_scheme=hyper['sampling_scheme'],
                        width=hyper['width'], height=hyper['height'],
                        attributes=hyper['attributes'], channels=hyper['channels'])
    train_dataloader = create_MR_structure(base_signal, hyper['max_stages'],hyper['filter'],hyper['decimation'])
    test_dataloader = create_MR_structure(base_signal, hyper['max_stages'],hyper['filter'])
    
    return train_dataloader, test_dataloader


def train_and_log(hyper, project_name, train_dataloader, test_dataloader):
    img_name = os.path.basename(hyper['image_name'])
    run_name = f"{img_name[0:5]}{hyper['model']}{hyper['filter'][0].upper()}"
    wandblogger = WandBLogger2D(project_name,
                            run_name,
                            hyper,
                            BASE_DIR,
                            visualize_gt_grads=hyper.get('visualize_grad', False))
    
    mrmodel = MRFactory.from_dict(hyper)
    print("Model: ", type(mrmodel))
    mrtrainer = MRTrainer.init_from_dict(mrmodel, train_dataloader, test_dataloader, wandblogger, hyper)
    mrtrainer.train(hyper['device'])


if __name__ == '__main__':
    project_name = "testing_periodic"
    #-- hyperparameters in configs --#
    config_file = 'configs/config_base_m_net.yml'
    with open(config_file) as f:
        hyper = yaml.load(f, Loader=SafeLoader)
        print(hyper)

    train_and_log(hyper, project_name, *create_dataloaders(hyper))
    