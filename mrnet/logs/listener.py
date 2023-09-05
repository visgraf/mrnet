import csv
import numpy as np
import os
import re
import torch
import wandb
import yaml

from scipy.fft import fft, fftfreq
from matplotlib import cm
from PIL import Image
from copy import deepcopy
from pathlib import Path
from typing import Sequence, Union
from torch.utils.data import BatchSampler

from mrnet.training.loss import gradient
from mrnet.datasets.sampler import make_grid_coords
from mrnet.datasets.utils import make_domain_slices

from mrnet.datasets.utils import (output_on_batched_dataset, 
                            output_on_batched_grid, rgb_to_grayscale, ycbcr_to_rgb, RESIZING_FILTERS, INVERSE_COLOR_MAPPING)
from mrnet.networks.mrnet import MRNet, MRFactory
from copy import deepcopy
import time
import trimesh
import skimage
from IPython import embed
from torchvision.transforms.functional import to_tensor, to_pil_image

MODELS_DIR = 'models'
MESHES_DIR = 'meshes'


def get_incremental_name(path):
    names = [nm.split()[0][-3:] for nm in os.listdir(path)]
    if not names:
        return 1
    names.sort()
    try:
        return int(names[-1]) + 1
    except ValueError:
        return 1


def make_runname(hyper, name):
    hyper = hyper
    stage = f"{hyper['stage']}-{hyper['max_stages']}"
    w0 = f"w{hyper['omega_0']}{'T' if hyper['superposition_w0'] else 'F'}"
    hl = f"hl{hyper['hidden_layers']}"
    epochs = f"MEp{hyper['max_epochs_per_stage']}"
    if isinstance(hyper['hidden_features'], Sequence):
        hf = ''.join([str(v) for v in hyper['hidden_features']])
    else:
        hf = hyper['hidden_features']
    hf = f"hf{hf}"
    res = f"r{hyper['width']}"
    per = f"pr{hyper['period']}"
    return f"{name}_{stage}_{w0}_{hf}_{epochs}_{hl}_{res}_{per}"

class BaseLogger:
    def __init__(self, project: str,
                        name: str, 
                        hyper: dict,
                        basedir: str, 
                        entity=None, 
                        config=None, 
                        settings=None):
        
        self.project = project
        self.name = name
        self.hyper = hyper
        self.basedir = basedir
        self.entity = entity
        self.config = config
        self.settings = settings

        try:
            self.runname = hyper['run_name']
        except KeyError:
            self.runname = make_runname(hyper, name)

    def log_images(self, pixels, label, captions=None, **kw):
        if not isinstance(pixels, Sequence):
            pixels = [pixels]
        if captions is None:
            captions = [None] * len(pixels)
        if isinstance(captions, str):
            captions = [captions]
        if len(pixels) != len(captions):
            raise ValueError("label and pixels should have the same size")
        
        try:
            # TODO: deal with color transform
            color_transform = INVERSE_COLOR_MAPPING[self.hyper.get(
                                                'color_space', 'RGB')]
            pixels = [color_transform(p.cpu()).clamp(0, 1) 
                    for p in pixels]
        except:
            pass
        return pixels, captions


class LocalLogger(BaseLogger):

    def make_dirs(self):
        for key in self.subpaths:
            os.makedirs(os.path.join(self.savedir, self.subpaths[key]), exist_ok=True)

    def __init__(self, project: str, 
                        name: str, 
                        hyper: dict,
                        basedir: str, 
                        entity=None, 
                        config=None, 
                        settings=None):
        super().__init__(project, name, hyper, 
                         basedir, entity, config, settings)
        self.logs = {}
        try: 
            logs_path = hyper['logs_path']
        except KeyError:
            logs_path = basedir.joinpath('logs')
        self.subpaths = {
            "models": "models",
            "gt": "gt",
            "pred": "pred",
            "etc": "etc",
            "zoom": "zoom"
        }
        
        self.savedir = os.path.join(logs_path, self.runname)

    def prepare(self, model):
        self.make_dirs()
        hypercontent = yaml.dump(self.hyper)
        with open(os.path.join(self.savedir, "hyper.yml"), "w") as hyperfile:
            hyperfile.write(hypercontent)

    def loglosses(self, log_dict:dict):
        filepath = os.path.join(self.savedir, 'losses.csv')
        file_exists = os.path.isfile(filepath)
        with open(filepath, 'a') as f:
            writer = csv.DictWriter(f, fieldnames=log_dict.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(log_dict)
            

    def log_images(self, pixels, label, captions=None, **kw):
        category = kw['category']
        pixels, captions = super().log_images(pixels, label, captions)
        path = os.path.join(self.savedir, self.subpaths[category])
        os.makedirs(path, exist_ok=True)

        for i, image in enumerate(pixels):
            try:
                filename = kw["fnames"][i]
            except (KeyError, IndexError):
                slug = '_'.join(label.lower().split())
                filename = f"{slug}{get_incremental_name(path):02d}"
            
            if captions:
                with open(os.path.join(path, 'captions.txt'), 'a') as capfile:
                    capfile.write(f"{filename}; {captions[i]}\n")
            
            filepath = os.path.join(path, filename + ".png")
            try:
                image.save(filepath)
            # if it is not a PIL Image
            except AttributeError:
                array = image.squeeze(-1).numpy()
                if array.dtype != np.uint8 and np.max(array) <= 1.0:
                    array = (array * 255).astype(np.uint8)
                Image.fromarray(array).save(filepath)
                

    def log_metric(self, metric_name, value, label):
        logpath = os.path.join(self.savedir, f"{metric_name}.csv")
        with open(logpath, "a") as metricfile:
            metricfile.write(f"{label}, {value}\n")

    def log_model(self, model, **kw):
        try:
            filename = kw['fname']
        except KeyError:
            filename = "final"
        logpath = os.path.join(self.savedir, self.subpaths['models'])
        os.makedirs(logpath, exist_ok=True)
        MRFactory.save(model, os.path.join(logpath, filename + ".pth"))

    def log_point_cloud(self, model, device):
        scale_radius = self.hyper.get('scale_radius', 0.9)
        if self.hyper.get('test_mesh', ""):
            try:
                mesh = trimesh.load_mesh(os.path.join(self.basedir, MESHES_DIR, 
                                                    self.hyper['test_mesh']))
                point_cloud, _ = trimesh.sample.sample_surface(
                                        mesh, self.hyper['ntestpoints'])
                point_cloud = torch.from_numpy(point_cloud).float()
                # center at the origin and rescale to fit a sphere of radius
                point_cloud = point_cloud - torch.mean(point_cloud, dim=0)
                scale = scale_radius / torch.max(torch.abs(point_cloud))
                point_cloud = scale * point_cloud
            except:
                point_cloud = torch.rand((self.hyper['ntestpoints'], 3))
                point_cloud = (point_cloud / torch.linalg.vector_norm(
                                    point_cloud, dim=-1).unsqueeze(-1)) * scale_radius
        else:
            point_cloud = torch.rand((self.hyper['ntestpoints'], 3))
            point_cloud = (point_cloud / torch.linalg.vector_norm(
                                    point_cloud, dim=-1).unsqueeze(-1)) * scale_radius
            
        with torch.no_grad():
            colors = output_on_batched_grid(model, 
                                            point_cloud, 
                                            self.hyper['batch_size'], 
                                            device).cpu().clamp(0, 1)
        if self.hyper['channels'] == 1:
            colors = torch.concat([colors, colors, colors], 1)
        else:
            color_transform = INVERSE_COLOR_MAPPING[self.hyper.get('color_space', 'RGB')]
            colors = color_transform(colors)
        colors = (colors * 255).int()
        point_cloud = torch.concat((point_cloud, colors), 1)

        wandb.log({"point_cloud": wandb.Object3D(point_cloud.numpy())})
    
    def finish(self):
        pass

class WandBLogger(BaseLogger):

    def prepare(self, model):
        wandb.finish()
        wandb.init(project=self.project, 
                    entity=self.entity, 
                    name=self.runname, 
                    config=self.hyper,
                    settings=self.settings)
        wandb.watch(model, log_freq=10, log='all')

    def loglosses(self, log_dict):
        wandb.log(log_dict)

    def log_images(self, pixels, label, captions=None, **kw):
        pixels, captions = super().log_images(pixels, label, captions)
        if isinstance(pixels[0], torch.Tensor):
            pixels = [p.squeeze(-1).numpy() for p in pixels]
        images = [wandb.Image(pixels[i],
                              caption=captions[i]) for i in range(len(pixels))]
        wandb.log({label: images})

    def log_metric(self, metric_name, value, label):
        table = wandb.Table(data=[(label, value)], 
                            columns = ["Stage", metric_name.upper()])
        wandb.log({
            f"{metric_name}_value" : wandb.plot.bar(
                                table, "Stage", metric_name.upper(),
                                title=f"{metric_name} for reconstruction")})
        
    def log_model(self, model, **kwargs):
        temp_path = kwargs['path']
        os.makedirs(temp_path, exist_ok=True)
        filename = kwargs.get('fname', 'final')
        logpath = os.path.join(temp_path, filename + '.pth')
        MRFactory.save(model, logpath)

        artifact = wandb.Artifact(filename, type='model')
        artifact.add_file(logpath)
        wandb.log_artifact(artifact)

    def log_pointcloud():
        pass

    def finish(self):
        wandb.finish()


class TrainingListener:

    def __init__(self, project: str, 
                        name: str, 
                        hyper: dict,
                        basedir: str, 
                        entity=None, 
                        config=None, 
                        settings=None) -> None:
        
        self.project = project
        self.name = name
        self.hyper = hyper
        self.basedir = basedir
        self.entity = entity
        self.config = config
        self.settings = settings

    def on_train_start(self):
        pass

    def on_stage_start(self, current_model, 
                       stage_number, updated_hyper=None):
        if updated_hyper:
            for key in updated_hyper:
                self.hyper[key] = updated_hyper[key]

        LoggerClass = (WandBLogger 
                       if self.hyper['logger'].lower() == 'wandb' 
                       else LocalLogger)
        self.logger = LoggerClass(self.project,
                                    self.name,
                                    self.hyper,
                                    self.basedir, 
                                    self.entity, 
                                    self.config, 
                                    self.settings)
        self.logger.prepare(current_model)

    def on_stage_trained(self, current_model: MRNet,
                                train_loader,
                                test_loader):
        device = self.hyper.get('eval_device', 'cpu')
        current_stage = current_model.n_stages()
        current_model.eval()
        current_model.to(device)
        
        start_time = time.time()
        if current_model.period > 0:
            self.log_chosen_frequencies(current_model)
        
        self.log_traindata(train_loader, stage=current_stage)
        gt = self.log_groundtruth(test_loader)
        pred = self.log_prediction(current_model, test_loader, device)
        self.log_PSNR(gt.cpu(), pred.cpu())
        self.log_SSIM(gt.cpu(), pred.cpu())

        try:
            self.log_extrapolation(current_model, 
                                   self.hyper['extrapolate'], 
                                   test_loader.size()[1:], 
                                   device)
        except KeyError:
            pass
        
        zoom = self.hyper.get('zoom', [])
        for zfactor in zoom:
            self.log_zoom(current_model, test_loader, zfactor, device)

        # TODO: check for pointcloud data
        
        print(f"[Logger] All inference done in {time.time() - start_time}s on {device}")
        current_model.train()
        current_model.to(self.hyper['device'])

    def on_batch_finish(self, batchloss):
        pass

    def on_epoch_finish(self, current_model, epochloss):
        log_dict = {f'{key.upper()} loss': value 
                        for key, value in epochloss.items()}
        if len(epochloss) > 1:
            log_dict['Total loss'] = sum(
                            [self.hyper['loss_weights'][k] * epochloss[k] 
                                for k in epochloss.keys()])

        self.logger.loglosses(log_dict)

    def on_train_finish(self, trained_model, total_epochs):
        self.log_model(trained_model)
        self.logger.finish()
        print(f'Total model parameters = ', trained_model.total_parameters())
        print(f'Training finished after {total_epochs} epochs')

    def log_model(self, model):
        temp_path = os.path.join(self.basedir, 'runs/tmp')
        self.logger.log_model(model, path=temp_path)
        
    def log_PSNR(self, gt, pred):
        mse = torch.mean((gt - pred)**2)
        psnr = 10 * torch.log10(1 / mse)
        
        label = f"Stage {self.hyper['stage']}"
        self.logger.log_metric("psnr", psnr, label)

    def log_SSIM(self, gt, pred):
        #clamped = pred.clamp(0, 1)
        transform = INVERSE_COLOR_MAPPING[self.hyper.get('color_space', 'RGB')]
        ssim = skimage.metrics.structural_similarity(
                        (transform(gt).cpu().numpy() * 255).astype(np.uint8), 
                        (transform(pred).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8),
                        data_range=1, channel_axis=-1)
        label = f"Stage {self.hyper['stage']}"
        self.logger.log_metric("ssim", ssim, label)
       
    def log_traindata(self, train_loader, **kw):
        pixels = train_loader.data.permute((1, 2, 0))
        print(pixels.shape, "DEBUG 1")
        if train_loader.domain_mask is not None:
            mask = train_loader.domain_mask.float().unsqueeze(-1)
            pixels = pixels * mask
        values = [pixels]
        captions = [f"{list(train_loader.shape)}"]
        
        self.logger.log_images(values, 'Train Data', captions, category='gt')
    
    def log_groundtruth(self, dataset):
        # permute to H x W x C
        pixels = dataset.data.permute((1, 2, 0))
        
        self.logger.log_images([pixels], 
                            'Ground Truth', 
                            [f"{list(dataset.shape)}"],
                            category='gt')
            
        color_space = self.hyper['color_space']
        if color_space == 'YCbCr':
            gray_pixels = pixels[..., 0]
        elif color_space == 'RGB':
            gray_pixels = rgb_to_grayscale(pixels).squeeze(-1)
        elif color_space == 'L':
            gray_pixels = pixels.squeeze(-1)
        else:
            raise ValueError(f"Invalid color space: {color_space}")
        
        self.log_fft(gray_pixels, 'FFT Ground Truth', category='gt')

        if 'd1' in self.hyper['attributes']:
            grads = dataset.data_attributes['d1']
            
            # TODO: move to log_gradmagnitude; deal with other color spaces
            if color_space == 'YCbCr':
                    grads = grads[0, ...]
            elif color_space == 'RGB':
                grads = (0.2126 * grads[0, ...] 
                        + 0.7152 * grads[1, ...] 
                        + 0.0722 * grads[2, ...])
            elif color_space == 'L':
                grads = grads.squeeze(0)
            
            self.log_gradmagnitude(grads, 'Gradient Magnitude GT', category='gt')
            
        return dataset.data.permute((1, 2, 0)
                                    ).reshape(-1, self.hyper['channels'])
    
    def log_prediction(self, model, test_loader, device):
        datashape = test_loader.shape[1:]
        
        coords = make_grid_coords(datashape, 
                                  *self.hyper['domain'],
                                  len(datashape))
        pixels = []
        grads = []
        color_space = self.hyper['color_space']
        for batch in BatchSampler(coords, 
                                  self.hyper['batch_size'], 
                                  drop_last=False):
            batch = torch.stack(batch)
            output_dict = model(batch.to(device))
            pixels.append(output_dict['model_out'].detach().cpu())
            value = output_dict['model_out']
            if color_space == 'YCbCr':
                value = value[:, 0:1]
            elif color_space == 'RGB':
                value = rgb_to_grayscale(value)

            grads.append(gradient(value, 
                                  output_dict['model_in']).detach().cpu())
            
        pixels = torch.concat(pixels)
        pred_pixels = pixels.reshape((*datashape, self.hyper['channels']))
        self.logger.log_images(pred_pixels, 'Prediction', category='pred')
        
        if color_space == 'YCbCr':
            gray_pixels = pred_pixels[..., 0]
        elif color_space == 'RGB':
            gray_pixels = rgb_to_grayscale(pred_pixels).squeeze(-1)
        elif color_space == 'L':
            gray_pixels = pred_pixels.squeeze(-1)
        else:
            raise ValueError(f"Invalid color space: {color_space}")
        
        self.log_fft(gray_pixels, 'FFT Prediction', category='pred')

        grads = torch.concat(grads)
        grads = grads.reshape((*datashape, 2))
        self.log_gradmagnitude(grads, 'Gradient Magnitude Pred', category='pred')
        return pixels
    
    def log_gradmagnitude(self, grads:torch.Tensor, label: str, **kw):
        mag = np.hypot(grads[:, :, 0].squeeze(-1).numpy(),
                        grads[:, :, 1].squeeze(-1).numpy())
        gmin, gmax = np.min(mag), np.max(mag)
        img = Image.fromarray(255 * (mag - gmin) / (gmax - gmin)).convert('L')
        self.logger.log_images([img], label, **kw)
    
    def log_fft(self, pixels:torch.Tensor, label:str, 
                    captions=None, **kw):
        '''Assumes a grayscale version of the image'''
        
        fft_pixels = torch.fft.fft2(pixels)
        fft_shifted = torch.fft.fftshift(fft_pixels).numpy()
        magnitude =  np.log(1 + abs(fft_shifted))
        # normalization to visualize as image
        vmin, vmax = np.min(magnitude), np.max(magnitude)
        magnitude = (magnitude - vmin) / (vmax - vmin)
        img = Image.fromarray((magnitude * 255).astype(np.uint8))
        self.logger.log_images([img], label, **kw)

    def log_extrapolation(self, model, interval, dims, device='cpu'):
        w, h = dims
        start, end = interval[0], interval[1]
        scale = (end - start) // 2
        neww, newh = int(scale * w), int(scale * h)
        
        ext_domain = make_grid_coords((neww, newh), start, end, dim=2)
        with torch.no_grad():
            pixels = output_on_batched_grid(model, 
                                            ext_domain, 
                                            self.hyper['batch_size'], 
                                            device)

        pixels = pixels.view((newh, neww, self.hyper['channels']))
        self.logger.log_images([pixels], 'Extrapolation', [f"{interval}"], category='pred')

    def log_zoom(self, model, test_loader, zoom_factor, device):
        w, h = test_loader.shape[1:]
        domain = self.hyper.get('domain', [-1, 1])
        start, end = domain[0]/zoom_factor, domain[1]/zoom_factor
        zoom_coords = make_grid_coords((w, h), start, end, dim=2)
        with torch.no_grad():
            pixels = output_on_batched_grid(model, zoom_coords, 
                                            self.hyper['batch_size'], device)

        pixels = pixels.cpu().view((h, w, self.hyper['channels']))
        if (self.hyper['channels'] == 1 
            and self.hyper['loss_weights']['d0'] == 0):
            vmin = torch.min(test_loader.data)
            vmax = torch.max(test_loader.data)
            pmin, pmax = torch.min(pixels), torch.max(pixels)
            pixels = (pixels - pmin) / (pmax - pmin)
            pixels = pixels * vmax #(vmax - vmin) + vmin
        
        # center crop
        cropsize = int(w // zoom_factor)
        left, top = (w - cropsize), (h - cropsize)
        right, bottom = (w + cropsize), (h + cropsize)
        crop_rectangle = tuple(np.array([left, top, right, bottom]) // 2)
        gt_pixels = test_loader.data.permute((1, 2, 0))
        
        color_space = self.hyper['color_space']
        color_transform = INVERSE_COLOR_MAPPING[color_space]
        pixels = color_transform(pixels).clamp(0, 1)
        gt_pixels = color_transform(gt_pixels).clamp(0, 1)

        pixels = (pixels.clamp(0, 1) * 255).squeeze(-1).numpy().astype(np.uint8)
        
        images = [Image.fromarray(pixels)]
        captions = [f'{zoom_factor}x Reconstruction (Ours)']
        fnames = [f'zoom_{zoom_factor}x_ours']
        gt_pixels = (gt_pixels * 255).squeeze(-1).numpy().astype(np.uint8)
        cropped = Image.fromarray(gt_pixels).crop(crop_rectangle)
        for filter in self.hyper.get('zoom_filters', ['linear']):
            resized = cropped.resize((w, h), RESIZING_FILTERS[filter])
            images.append(resized)
            captions.append(f"{zoom_factor}x Baseline - {filter} interpolation")
            fnames.append(f'zoom_{zoom_factor}x_{filter}')
             
        self.logger.log_images(images, f"Zoom {zoom_factor}x", captions, 
                                    fnames=fnames, category='zoom')

    def log_chosen_frequencies(self, model: MRNet):
        # changes with dimensions
        frequencies = []
        for stage in model.stages:
            last_stage_frequencies = stage.first_layer.linear.weight
            frequencies.append(last_stage_frequencies)
        frequencies = torch.cat(frequencies).cpu().numpy()
        frequencies = (frequencies * model.period 
                       / (2 * np.pi)).astype(np.int32)
        h, w = self.hyper['width'], self.hyper['height']
        frequencies = frequencies + np.array((h//2, w//2))
        img = Image.new('L', (h, w))
        for f in frequencies:
            img.putpixel(f, 255)
        
        self.logger.log_images([img], "Chosen Frequencies", category='etc')