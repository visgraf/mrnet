import os
from pathlib import Path
import torch

from mrnet.training.trainer import MRTrainer
from mrnet.datasets.signals import ImageSignal
from mrnet.networks.mrnet import MRFactory
from mrnet.datasets.pyramids import create_MR_structure
from mrnet.logs.listener import TrainingListener


from utils import load_hyperparameters, get_optim_handler


BASE_DIR = Path('.').absolute()
IMAGE_PATH = BASE_DIR.joinpath('img')
MODEL_PATH = BASE_DIR.joinpath('models')
torch.manual_seed(777)

#-- hyperparameters in configs --#

if __name__ == '__main__':
    hyper = load_hyperparameters('docs/configs/image.yml')
    imgpath = hyper['image_name']
    project_name = hyper.get('project_name', 'dev_sandbox')

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

    # you can substitute this line by your custom handler class
    optim_handler = get_optim_handler(hyper)

    img_name = os.path.basename(hyper['image_name'])
    mrmodel = MRFactory.from_dict(hyper)
    print("Model: ", type(mrmodel))

    wandblogger = TrainingListener(project_name,
                                f"{hyper['model']}{hyper['filter'][0].upper()}{img_name[0:5]}{hyper['color_space'][0]}",
                                hyper,
                                Path("runs"))

    mrtrainer = MRTrainer.init_from_dict(mrmodel,
                                        train_dataset,
                                        test_dataset,
                                        wandblogger,
                                        hyper,
                                        optim_handler=optim_handler)
    mrtrainer.train(hyper['device'])