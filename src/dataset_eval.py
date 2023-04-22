import torch
import os
from pathlib import Path
from logs.locallogger import LocalLogger2D, LocalLogger
from training.trainer import MRTrainer
from datasets.imagesignal import ImageSignal
from networks.mrnet import MRFactory
from datasets.pyramids import create_MR_structure
from datasets.sampler import make_grid_coords
import yaml
from yaml.loader import SafeLoader
import matplotlib.pyplot as plt
from PIL import Image

def get_base_dir():
    base = Path('.').absolute()
    children = os.listdir(base)
    if 'src' in children and 'docs' in children:
        return base
    return Path('.').absolute().parents[0]

def center_crop(img, newsize):
    width, height = img.size   # Get dimensions
    left = (width - newsize) // 2
    top = (height - newsize) // 2
    right = (width + newsize) // 2
    bottom = (height + newsize) // 2
    # Crop the center of the image
    return img.crop((left, top, right, bottom))

def prepare_dataset(datapath, newsize):
    filenames = os.listdir(datapath)
    for i, name in enumerate(filenames):
        filepath = os.path.join(datapath, name)
        img = Image.open(filepath)
        img = center_crop(img, newsize)
        img.save(filepath)
        print(f"{i}. Processed: ", filepath)

def run_experiment(project_name, dataset_relpath, configfile):
    base_dir = get_base_dir()
    logs_path = base_dir.joinpath('logs')
    models_path = os.path.join(logs_path, project_name, 'models')
    DATASET_PATH = os.path.join(base_dir, dataset_relpath)

    config_file = os.path.join(base_dir, 'configs', configfile)
    with open(config_file) as f:
        hyper = yaml.load(f, Loader=SafeLoader)
        print(hyper)

    filenames = os.listdir(DATASET_PATH)
    os.makedirs(models_path, exist_ok=True)

    for name in filenames:
        hyper['image_name'] = name
        base_signal = ImageSignal.init_fromfile(
                            os.path.join(DATASET_PATH, hyper['image_name']),
                            batch_samples_perc=hyper['batch_samples_perc'],
                            sampling_scheme=hyper['sampling_scheme'],
                            width=hyper['width'], height= hyper['height'],
                            attributes=hyper['attributes'],
                            channels=hyper['channels'])
        train_dataloader = create_MR_structure(base_signal, hyper['max_stages'],
                                                hyper['filter'], hyper['decimation'])
        test_dataloader = create_MR_structure(base_signal, hyper['max_stages'],
                                                hyper['filter'])

        locallogger = LocalLogger2D(project_name,
                                    f"{hyper['model']}net{hyper['max_stages']}Stg{hyper['hidden_features'][0]}B{'T' if hyper['decimation'] else 'F'}",
                                    hyper,
                                    base_dir, 
                                    to_file=True)
        mrmodel = MRFactory.from_dict(hyper)
        print("Model: ", type(mrmodel))
        mrtrainer = MRTrainer.init_from_dict(mrmodel, 
                                            train_dataloader, test_dataloader, locallogger, hyper)
        mrtrainer.train(hyper['device'])


        filename = f"{hyper['model']}{hyper['image_name'][0:4]}.pth"
        path = os.path.join(models_path, filename)

        MRFactory.save(mrmodel, path)
        

if __name__ == '__main__':
    # prepare_dataset('E:\Workspace\impa\mrimg\img\kodak512', 512)
    # run_experiment('kodak512', 'img/kodak512', 'config_kodak_m_net.yml')
    run_experiment('kodak512', 'img/kodak512', 'config_kodak_siren.yml') 