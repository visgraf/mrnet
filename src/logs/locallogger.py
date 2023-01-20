from networks.mrnet import MRNet
from .logger import Logger

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
from training.loss import gradient
from datasets.imagesignal import ImageSignal
from typing import Sequence, Tuple
import yaml

from IPython import embed

class LocalLogger(Logger):
    def __init__(self, project: str, 
                        name: str, 
                        hyper: dict,
                        basedir: str,
                        to_file: bool = False, 
                        entity = None, config=None, settings=None):
        self.project = project
        self.name = name
        self.hyper = hyper
        self.basedir = basedir
        self.to_file = to_file
        self.entity = entity
        self.config = config
        self.settings = settings
        self.logs = {}
        if self.to_file:
            self.savedir = os.path.join(basedir, 'logs', project, name)
            os.makedirs(self.savedir, exist_ok=True)
            os.makedirs(os.path.join(self.savedir, "gt"), exist_ok=True)
            os.makedirs(os.path.join(self.savedir, "pred"), exist_ok=True)
            hypercontent = yaml.dump(self.hyper)
            with open(os.path.join(self.savedir, "hyper.yml"), "w") as hyperfile:
                hyperfile.write(hypercontent)


class LocalLogger2D(LocalLogger):

    def on_train_start(self):
        self.trainloss = []

    def on_epoch_finish(self, current_model, epochloss):
        self.trainloss.append(sum(epochloss.values()))

    def on_stage_start(self, current_model, stage_number, updated_hyper=None):
        print(f'Stage {stage_number} starting')
        print("STARTED STAGE", stage_number)
        self.logs[stage_number] = {}

    def on_stage_trained(self, current_model: MRNet,
                                train_loader: DataLoader,
                                test_loader: DataLoader):
        device = self.hyper.get('eval_device', 'cpu')
        current_model.eval()
        current_model.to(device)
        print(current_model.top_stage)
        current_model.train()

        # log groundtruth
        stage_number = current_model.n_stages()
        gt = self.log_groundtruth(test_loader, stage_number)
        pred = self.log_prediction(test_loader, current_model, 
                            device, stage_number)
                            
        self.log_PSNR(gt.to(device), pred, stage_number)
        
    def on_train_finish(self, trained_model, total_epochs):
        if self.to_file:
            filename = os.path.join(self.savedir, "psnr.csv")
            psnr = self.logs[self.hyper['max_stages']]['psnr']
            with open(filename, "a") as psnrfile:
                psnrfile.write(f"{os.path.basename(self.hyper['image_name'])}, {psnr}\n")
        print(f'Training finished after {total_epochs} epochs')

    def tensor_to_img(self, tensor):
        data = tensor.view(-1, self.hyper['channels'])
        w = self.hyper['width']
        h = self.hyper['height']
        c = self.hyper['channels']
        pixels = data.cpu().detach().view(w, h, c).numpy()
        pixels = (pixels * 255).astype(np.uint8)
        if c == 1:
            pixels = np.repeat(pixels, 3, axis=-1)
        print("CHANNELS", self.hyper['channels'])
        return Image.fromarray(pixels)

    def log_groundtruth(self, test_loader, stage_number):
        img = self.tensor_to_img(test_loader.data)
        if self.to_file:
            filename = os.path.join(self.savedir, "gt", f"gt{stage_number}_{os.path.basename(self.hyper['image_name'])}")
            print("FILENAME", filename)
            img.save(filename)
            print("SAVED GT")
        else:
            self.logs[stage_number]['gt'] = img
        return test_loader.data.view(-1, self.hyper['channels'])

    def log_prediction(self, test_loader, model, device, stage_number):
        output_dict = model(test_loader.sampler.coords_vis.to(device))
        model_out = torch.clamp(output_dict['model_out'], 0.0, 1.0)

        img = self.tensor_to_img(model_out)
        if self.to_file:
            filename = os.path.join(self.savedir, "pred", f"pred{stage_number}_{os.path.basename(self.hyper['image_name'])}")
            img.save(filename)
        else:
            self.logs[stage_number]['pred'] = img
        
        return model_out

    def log_PSNR(self, gt, pred, stage_number):
        psnr = 10*torch.log10(1 / (torch.mean(gt - pred)**2 + 1e-10))
        print(self.logs)
        self.logs[stage_number]['psnr'] = psnr