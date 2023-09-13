import os
from pathlib import Path
import torch

from mrnet.training.trainer import MRTrainer
from mrnet.datasets.signals import ImageSignal
from mrnet.networks.mrnet import MRFactory
from mrnet.datasets.pyramids import create_MR_structure
from mrnet.training.listener import TrainingListener

from utils import load_hyperparameters, get_optim_handler


def make_triangle_mask(res):
    mask = torch.zeros(res, res, dtype=torch.bool)
    for i in range(len(mask)):
        for j in range(len(mask[0])):
            if i > j:
                mask[i, j] = True
    return mask

if __name__ == '__main__':
    torch.manual_seed(777)
    #-- hyperparameters in configs --#
    hyper = load_hyperparameters('docs/configs/masked.yml')
    project_name = hyper.get('project_name', 'framework-tests')

    base_signal = ImageSignal.init_fromfile(
                        hyper['data_path'],
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
    # masks only tested in single resolution at this time
    mask = make_triangle_mask(hyper['width'])
    base_signal.add_mask(mask)
    train_dataset = [base_signal]
    test_dataset = [base_signal]

    # you can substitute this line by your custom handler class
    optim_handler = get_optim_handler(hyper.get('optim_handler', 'regular'))

    mrmodel = MRFactory.from_dict(hyper)
    print("Model: ", type(mrmodel))
    name = os.path.basename(hyper['data_path'])
    logger = TrainingListener(project_name,
                                f"{hyper['model']}{hyper['filter'][0].upper()}{name[0:7]}{hyper['color_space'][0]}",
                                hyper,
                                Path(hyper.get("log_path", "runs")))
    mrtrainer = MRTrainer.init_from_dict(mrmodel,
                                        train_dataset,
                                        test_dataset,
                                        logger,
                                        hyper,
                                        optim_handler=optim_handler)
    mrtrainer.train(hyper['device'])