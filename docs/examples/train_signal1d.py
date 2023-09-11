import os
from pathlib import Path
import torch

from mrnet.training.trainer import MRTrainer
from mrnet.datasets.signals import Signal1D
from mrnet.networks.mrnet import MRFactory
from mrnet.datasets.pyramids import create_MR_structure
from mrnet.logs.listener import TrainingListener
from mrnet.datasets.procedural import perlin_noise

from utils import load_hyperparameters, get_optim_handler


BASE_DIR = Path('.').absolute()
DATA_PATH = BASE_DIR.joinpath('data')
torch.manual_seed(777)
# os.environ['WANDB_MODE'] = "offline"

#-- hyperparameters in configs --#

def proc_noise(hyper):
    scale = hyper.get('scale', 10) 
    octaves = hyper.get('octaves', 1)
    p = hyper.get('p', 1)
    def inner(nsamples):
        return perlin_noise(nsamples, scale, octaves, p)
    return inner

if __name__ == '__main__':
    hyper = load_hyperparameters('docs/configs/signal1d.yml')
    project_name = hyper.get('project_name', 'dev_sandbox')

    # procedure = proc_noise(hyper)
    base_signal = Signal1D.init_fromfile(
                        hyper['data_path'],
                        domain=hyper['domain'],
                        sampling_scheme=hyper['sampling_scheme'],
                        attributes=hyper['attributes'],
                        batch_size=hyper['batch_size'])
    
    hyper['nsamples'] = len(base_signal.coords)

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

    # you can substitute this line by your custom handler class
    optim_handler = get_optim_handler(hyper)

    try:
        name = os.path.basename(hyper['data_path'])
    except KeyError:
        name = 'procedural'

    mrmodel = MRFactory.from_dict(hyper)
    print("Model: ", type(mrmodel))

    training_listener = TrainingListener(project_name,
                                f"{hyper['model']}{hyper['filter'][0].upper()}{name[0:5]}",
                                hyper,
                                Path("runs"))

    mrtrainer = MRTrainer.init_from_dict(mrmodel,
                                        train_dataset,
                                        test_dataset,
                                        training_listener,
                                        hyper,
                                        optim_handler=optim_handler)
    mrtrainer.train(hyper['device'])