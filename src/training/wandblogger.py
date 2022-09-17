import wandb
import torch
import os
import re
import numpy as np
import torch.utils.data as data_utils

from torch.utils.data import DataLoader
from scipy.fft import fft, fftfreq
from matplotlib import cm
from PIL import Image
from copy import deepcopy
from pathlib import Path

import warnings
from training.loss import gradient
from datasets.sampling import make2Dcoords

from .logger import Logger
from networks.mrnet import MRNet, MRFactory

MODELS_DIR = 'models'

class WandBLogger(Logger):
    def __init__(self, project: str, 
                        name: str, 
                        hyper: dict,
                        basedir: str, 
                        entity = None, config=None, settings=None):
        self.project = project
        self.name = name
        self.hyper = hyper
        self.basedir = basedir
        self.entity = entity
        self.config = config
        self.settings = settings

    def on_epoch_finish(self, current_model, epochloss):
        log_dict = {f'{key.upper()} loss': value 
                        for key, value in epochloss.items()}
        if len(epochloss) > 1:
            log_dict['Total loss'] = sum(epochloss.values())

        wandb.log(log_dict)

    def on_train_finish(self, trained_model, total_epochs):
        save_format = self.hyper.get('save_format', None)
        if save_format is not None:
            self.log_model(trained_model, save_format)
        
        wandb.finish()
        print(f'Total model parameters = ', trained_model.total_parameters())
        print(f'Training finished after {total_epochs} epochs')

##

class WandBLogger2D(WandBLogger):
    
    def on_stage_start(self, current_model, stage_number, updated_hyper=None):
        if updated_hyper:
            for key in updated_hyper:
                self.hyper[key] = updated_hyper[key]
        
        hyper = self.hyper
        self.runname = f"{self.name}_{hyper['stage']}/{hyper['max_stages']}_w{hyper['omega_0']}{'T' if hyper['superposition_w0'] else 'F'}_hf{hyper['hidden_features']}_MEp{hyper['max_epochs_per_stage']}_hl{hyper['hidden_layers']}_{hyper['width']}px"
        wandb.init(project=self.project, 
                    entity=self.entity, 
                    name=self.runname, 
                    config=self.hyper,
                    settings=self.settings)
        wandb.watch(current_model, log_freq=10, log='all')


    def on_stage_trained(self, current_model: MRNet,
                                train_loader: DataLoader,
                                test_loader: DataLoader):
        device = self.hyper.get('eval_device', 'cpu')
        current_model.eval()
        current_model.to(device)

        self.log_traindata(train_loader)
        gt = self.log_groundtruth(test_loader)        
        pred = self.log_prediction(current_model, test_loader, device)
        self.log_PSNR(gt.to(device), pred)

        extrapolation_interval = self.hyper.get('extrapolate', None)
        if extrapolation_interval is not None:
            self.log_extrapolation(current_model, extrapolation_interval, 
                                    test_loader.dataset.dimensions(), device)

        current_model.train()
        current_model.to(self.hyper['device'])
       
##
    def log_traindata(self, train_loader):
        traindata = train_loader.dataset.data.view(-1, self.hyper['channels'])
        pixels = self.as_imagetensor(torch.clamp(traindata, 0, 1))

        if re.match('laplace_*', self.hyper['multiresolution']) and self.hyper['stage'] > 1:
            self.log_detailtensor(pixels, 'Train Data')
        else:
            self.log_imagetensor(pixels, 'Train Data')
    
    def log_groundtruth(self, test_loader):
        gtdata = test_loader.dataset.data.view(-1, self.hyper['channels'])
        pixels = self.as_imagetensor(torch.clamp(gtdata, 0, 1))
        
        if re.match('laplace_*', self.hyper['multiresolution']) and self.hyper['stage'] > 1:
            self.log_detailtensor(pixels, 'Ground Truth')
        else:
            self.log_imagetensor(pixels, 'Ground Truth')
        
        self.log_fft(pixels, 'FFT Ground Truth')

        if self.hyper.get('useattributes', False):
            gt_grads = test_loader.dataset.d1
            self.log_gradmagnitude(gt_grads, 'Ground Truth')
        return gtdata

    def log_prediction(self, model, test_loader, device):
        output_dict = model(test_loader.dataset.coordinates.to(device))
        model_out = torch.clamp(output_dict['model_out'], 0.0, 1.0)

        pred_pixels = self.as_imagetensor(model_out)
        self.log_imagetensor(pred_pixels, 'Prediction')
        self.log_fft(pred_pixels, 'FFT Prediction')

        if self.hyper.get('useattributes', False):
            model_grads = gradient(model_out, output_dict['model_in'])
            pred_grads = torch.reshape(model_grads, (-1, 2))
            self.log_gradmagnitude(pred_grads, 'Prediction')
        
        return model_out
    
    # TODO: make it work with color images and non-squared images

    def as_imagetensor(self, tensor):
        w = h = int(np.sqrt(len(tensor)))
        pixels = tensor.cpu().detach().unflatten(0, (w, h))
        return pixels

    def log_imagetensor(self, pixels:torch.Tensor, label:str):
        image = wandb.Image(pixels.numpy())    
        wandb.log({label: image})

    def log_detailtensor(self, pixels:torch.Tensor, label:str):
        #imin, imax = torch.min(pixels), torch.max(pixels)
        pixels = (pixels + 1.0) / (2.0)
        image = wandb.Image(pixels.numpy())    
        wandb.log({label: image})
    
    def log_gradmagnitude(self, grads:torch.Tensor, label: str):
        grads = self.as_imagetensor(grads)
        mag = np.hypot(grads[:, :, 0].squeeze(-1).numpy(),
                        grads[:, :, 1].squeeze(-1).numpy())
        gmin, gmax = np.min(mag), np.max(mag)
        img = Image.fromarray(255 * (mag - gmin) / gmax).convert('L')
        wandb.log({f'Gradient Magnitude - {label}': wandb.Image(img)})
    
    def log_fft(self, pixels:torch.Tensor, label:str):
        fourier_tensor = torch.fft.fftshift(
                        torch.fft.fft2(pixels.squeeze(-1)))
        magnitude = 20 * np.log(abs(fourier_tensor.numpy()) + 1e-10)
        magnitude = magnitude / np.max(magnitude)
        graymap = cm.get_cmap('gray')
        img = Image.fromarray(np.uint8(graymap(magnitude) * 255))
        wandb.log({label: wandb.Image(img)})

    def log_PSNR(self, gt, pred):
        psnr = 10*torch.log10(1 / (torch.mean(gt - pred)**2 + 1e-10))
        
        label = f"Stage {self.hyper['stage']}"
        table = wandb.Table(data=[(label, psnr)], columns = ["Stage", "PSNR"])
        wandb.log({"psnr_value" : wandb.plot.bar(table, "Stage", "PSNR",
                                    title="PSNR for reconstruction")})

    def log_extrapolation(self, model, interval, dims, device='cpu'):
        w, h = dims
        space = 2.0 / w
        start, end = interval[0], interval[1]
        newsamplesize = abs(int((end - start) / space)) + 1
        ext_domain = make2Dcoords(newsamplesize, newsamplesize, start, end)

        output_dict = model(ext_domain.to(device))
        model_out = torch.clamp(output_dict['model_out'].detach(), 0, 1)

        pixels = self.as_imagetensor(model_out)
        self.log_imagetensor(pixels, 'Extrapolation')

    def log_model(self, model, save_format):
        filename = f"{self.runname}.pth".replace('/', '-')
        mdir = os.path.join(self.basedir, MODELS_DIR)
        Path(mdir).mkdir(parents=True, exist_ok=True)
        path = os.path.join(mdir, filename)
        if save_format == 'script':
            scripted = torch.jit.script(model)
            scripted.save(path)
        # TODO: check if it should be moved to a network class
        elif save_format == 'general':
            MRFactory.save(model, path)
        else:
            warnings.warn(
                f'Model not saved! Save type {save_format} NOT supported.')
            return None
        print("File ", filename)
        artifact = wandb.Artifact(filename, type='model')
        artifact.add_file(path)
        wandb.log_artifact(artifact)