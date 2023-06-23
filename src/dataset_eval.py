import torch
import os
from pathlib import Path
from logs.locallogger import LocalLogger2D
from logs.wandblogger import WandBLogger2D
from training.trainer import MRTrainer
from datasets.signals import ImageSignal
from networks.mrnet import MRFactory
from datasets.pyramids import create_MR_structure
from datasets.sampler import make_grid_coords
import yaml
from yaml.loader import SafeLoader
import matplotlib.pyplot as plt
from PIL import Image
from PIL.Image import Resampling

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

def prepare_dataset(datapath, newsize, resize=False, rename_to=""):
    filenames = os.listdir(datapath)
    print(f"{len(filenames)} files found at {datapath}")
    print(filenames)
    for i, name in enumerate(filenames):
        filepath = os.path.join(datapath, name)
        img = Image.open(filepath)
        if resize:
            w, h = img.size
            if w < h:
                s = int(newsize * h/w)
                img = img.resize((newsize, s), Resampling.BICUBIC)
            else:
                s = int(newsize * w/h)
                img = img.resize((s, newsize), Resampling.BICUBIC)

        img = center_crop(img, newsize)
        if rename_to:
            newpath = os.path.join(datapath, f'{rename_to}{i}.png')
            img.save(newpath)
            os.remove(filepath)
        else:
            img.save(filepath)
        print(f"{i}. Processed: ", filepath)

def run_experiment(project_name, dataset_relpath, configfile, LoggerClass):
    base_dir = get_base_dir()
    logs_path = base_dir.joinpath('logs')
    models_path = os.path.join(logs_path, project_name, 'models')
    DATASET_PATH = os.path.join(base_dir, dataset_relpath)

    config_file = os.path.join(base_dir, 'configs/siggraph_asia', configfile)

    filenames = os.listdir(DATASET_PATH)
    os.makedirs(models_path, exist_ok=True)
    print(len(filenames), "imagens")
    for name in filenames:
        torch.manual_seed(777)
        with open(config_file) as f:
            hyper = yaml.load(f, Loader=SafeLoader)
            print(hyper)
            if isinstance(hyper['batch_size'], str):
                hyper['batch_size'] = eval(hyper['batch_size'])
            if hyper['channels'] == 0:
                hyper['channels'] = hyper['out_features']
            print(hyper)
        hyper['image_name'] = name
        imgpath = os.path.join(DATASET_PATH, hyper['image_name'])
        print(imgpath)
        base_signal = ImageSignal.init_fromfile(
                    imgpath,
                    domain=hyper['domain'],
                    channels=hyper['channels'],
                    sampling_scheme=hyper['sampling_scheme'],
                    attributes=hyper['attributes'],
                    batch_size=hyper['batch_size'],
                    color_space=hyper['color_space'])
        hyper['width'], hyper['height'] = base_signal.shape[1:]

        train_dataloader = create_MR_structure(base_signal, 
                                               hyper['max_stages'],
                                                hyper['filter'], 
                                                hyper['decimation'])
        test_dataloader = create_MR_structure(base_signal, 
                                              hyper['max_stages'],
                                              hyper['filter'], False)

        logger = LoggerClass(project_name,
                                    f"{hyper['model']}{hyper['filter'][0].upper()}{name[0:5]}{hyper['color_space'][0]}",
                                    hyper,
                                    base_dir) 
        mrmodel = MRFactory.from_dict(hyper)
        print("Model: ", type(mrmodel))
        mrtrainer = MRTrainer.init_from_dict(mrmodel, 
                                            train_dataloader, 
                                            test_dataloader, 
                                            logger, hyper)
        mrtrainer.train(hyper['device'])
        print("DONE", name)


if __name__ == '__main__':
    run_experiment('siggraph_asia_siren', 
                   'img/allin', 
                   'config_siren.yml',
                   WandBLogger2D)
    
    # run_experiment('siggraph_asia', 
    #                'img/siggraph_asia', 
    #                'config_siggraph_imgs_rgb.yml',
    #                WandBLogger2D)

    # prepare_dataset('E:\Workspace\impa\mrnet\img\pexels_textures', 1024, 
    #                 resize=True, rename_to="pic")
    # run_experiment('pexels1024', 'img/pexels_textures', 
    #                'config_textures_m_net.yml', WandBLogger2D)
    # run_experiment('pexels1024', 'img/pexels_textures', 
    #                'config_textures_l_net.yml', WandBLogger2D)
    # run_experiment('pexels1024', 'img/pexels_textures', 
    #                'config_textures_m_net_color.yml', WandBLogger2D)
    # run_local_experiment('kodak512', 'img/kodak512', 'config_kodak_siren.yml')