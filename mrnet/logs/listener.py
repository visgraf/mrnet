import wandb
import torch
import os
import re
import numpy as np
import yaml

from scipy.fft import fft, fftfreq
from matplotlib import cm
from PIL import Image
from copy import deepcopy
from pathlib import Path
from typing import Sequence, Union
from torch.utils.data import BatchSampler

import warnings
from mrnet.training.loss import gradient
from mrnet.datasets.sampler import make_grid_coords
from mrnet.datasets.utils import make_domain_slices

# from .logger import Logger
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

    def log_images(self, pixels, category, label, captions):
        if not isinstance(pixels, Sequence):
            pixels = [pixels]
        if captions is None:
            captions = [None] * len(pixels)
        if not isinstance(captions, Sequence):
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

    # def log_image(self, image, label):
    #     if "ground truth" in label.lower():
    #         path = os.path.join(self.savedir, self.subpaths['gt'])
    #     else:
    #         path = os.path.join(self.savedir, self.subpaths['pred'])
    #     image.save(os.path.join(path, label), bitmap_format='png')

    def loglosses(self, log_dict):
        # wandb.log(log_dict)
        # TODO: log losses
        pass

    def log_images(self, pixels, category, label, captions=None):
        super().log_images(pixels, category, label, captions)
        path = os.path.join(self.savedir, self.subpaths[category])
        os.makedirs(path, exist_ok=True)

        for i, image in enumerate(pixels):
            filename = (captions[i] if captions
                        else f"{get_incremental_name(path):02d}")
            try:
                image.save(filename, bitmap_format='png')
            # if it is not a PIL Image
            except AttributeError:   
                image = Image.fromarray(image.squeeze(-1).numpy())
                image.save(filename, bitmap_format='png')

    def log_metric(self, metric_name, value, label):
        logpath = os.path.join(self.basepath, f"{metric_name}.csv")
        with open(logpath, "a") as metricfile:
            metricfile.write(f"{label}, {value}")

    def log_model(self, model, filename, **kwargs):
        logpath = os.path.join(self.basepath, filename)
        os.makedirs(logpath, exist_ok=True)
        MRFactory.save(model, logpath)

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

class WandBLogger:

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

    def log_image(image, label):
        wandb.log({label: wandb.Image(image)})

    def log_images(pixels, label, captions):
        images = [wandb.Image(pixels[i].squeeze(-1).numpy(),
                              caption=captions[i]) for i in range(len(pixels))]
        wandb.log({label: images})

    def log_metric(self, metric_name, value, label):
        table = wandb.Table(data=[(label, value)], 
                            columns = ["Stage", metric_name.upper()])
        wandb.log({
            f"{metric_name}_value" : wandb.plot.bar(
                                table, "Stage", metric_name.upper(),
                                title=f"{metric_name} for reconstruction")})
        
    def log_model(self, model, filename, **kwargs):
        path = kwargs['path']
        logpath = os.path.join(path, filename)
        MRFactory.save(model, logpath)

        artifact = wandb.Artifact(filename, type='model')
        artifact.add_file(logpath)
        wandb.log_artifact(artifact)

    def log_pointcloud():
        pass

    def log_zoom():
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
        self.logger = LocalLogger(self.project,
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
        current_model.eval()
        current_model.to(device)
        
        start_time = time.time()
        if current_model.period > 0:
            self.log_chosen_frequencies(current_model)
        
        self.log_traindata(train_loader)
        gt = self.log_groundtruth(test_loader)
        pred = self.log_prediction(current_model, test_loader, device)
        
        # rewritten
        self.log_PSNR(gt.cpu(), pred.cpu())
        self.log_SSIM(gt.cpu(), pred.cpu())
        # ---------
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
        self.logger.finish()
        print(f'Total model parameters = ', trained_model.total_parameters())
        print(f'Training finished after {total_epochs} epochs')

    def log_model(self, model, save_format):
        filename = f"{self.runname}.pth".replace('/', '-')
        modelpath = os.path.join(self.basedir, MODELS_DIR)
        Path(modelpath).mkdir(parents=True, exist_ok=True)

        self.logger.log_model(model, filename, path=modelpath)
        
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
        
    # def log_images(self, pixels, label, captions=None):
    #     if isinstance(pixels, torch.Tensor):
    #         pixels = [pixels]
    #     if captions is None:
    #         captions = [None] * len(pixels)
    #     if isinstance(captions, str):
    #         captions = [captions]
    #     if len(pixels) != len(captions):
    #         raise ValueError("label and pixels should have the same size")
        
    #     color_transform = INVERSE_COLOR_MAPPING[self.hyper.get(
    #                                             'color_space', 'RGB')]
    #     pixels = [color_transform(p.cpu()).clamp(0, 1) 
    #               for p in pixels]
        
    #     self.logger.log_images(pixels, label, captions)
       
    def log_traindata(self, train_loader):
        pixels = train_loader.data.permute((1, 2, 0))
        if train_loader.domain_mask is not None:
            mask = train_loader.domain_mask.float().unsqueeze(-1)
            pixels = pixels * mask
            values = [pixels]
            captions = [f"{list(train_loader.shape)}"]
        else: 
            values = [pixels]
            captions = [f"{list(train_loader.shape)}"]
        self.logger.log_images(values, 'gt', 'Train Data', captions)
    
    def log_groundtruth(self, dataset):
        # permute to H x W x C
        pixels = dataset.data.permute((1, 2, 0))
        
        self.logger.log_images([pixels], 'gt', 
                            'Ground Truth', 
                            [f"{list(dataset.shape)}"])
            
        color_space = self.hyper['color_space']
        if color_space == 'YCbCr':
            gray_pixels = pixels[..., 0]
        elif color_space == 'RGB':
            gray_pixels = rgb_to_grayscale(pixels).squeeze(-1)
        elif color_space == 'L':
            gray_pixels = pixels.squeeze(-1)
        else:
            raise ValueError(f"Invalid color space: {color_space}")
        
        self.log_fft(gray_pixels, 'gt', 'FFT Ground Truth')

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
            
            self.log_gradmagnitude(grads, 'gt', 'GT ')
            # mag = np.hypot(grads[:, :, 0].squeeze(-1).numpy(),
            #                grads[:, :, 1].squeeze(-1).numpy())
            # vmin, vmax = np.min(mag), np.max(mag)
            # img = Image.fromarray(255 * (mag - vmin) / (vmax - vmin)).convert('L')
            # wandb.log({f'Gradient Magnitude - {"GT"}': wandb.Image(img)})
            
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
        self.log_images(pred_pixels, 'Prediction')
        
        
        if color_space == 'YCbCr':
            gray_pixels = pred_pixels[..., 0]
        elif color_space == 'RGB':
            gray_pixels = rgb_to_grayscale(pred_pixels).squeeze(-1)
        elif color_space == 'L':
            gray_pixels = pred_pixels.squeeze(-1)
        else:
            raise ValueError(f"Invalid color space: {color_space}")
        
        self.log_fft(gray_pixels, 'FFT Prediction')

        grads = torch.concat(grads)
        # model_grads = gradient(model_out, output_dict['model_in'])
        grads = grads.reshape((*datashape, 2))
        self.log_gradmagnitude(grads, 'Prediction - Gradient')
        return pixels
    
    def log_gradmagnitude(self, grads:torch.Tensor, category, label: str):
        mag = np.hypot(grads[:, :, 0].squeeze(-1).numpy(),
                        grads[:, :, 1].squeeze(-1).numpy())
        gmin, gmax = np.min(mag), np.max(mag)
        img = Image.fromarray(255 * (mag - gmin) / (gmax - gmin)).convert('L')
        # wandb.log({f'Gradient Magnitude - {label}': wandb.Image(img)})
        self.logger.log_images(img, category, label)
    
    def log_fft(self, pixels:torch.Tensor, category:str, 
                label:str, captions=None):
        '''Assumes a grayscale version of the image'''
        
        fft_pixels = torch.fft.fft2(pixels)
        fft_shifted = torch.fft.fftshift(fft_pixels).numpy()
        magnitude =  np.log(1 + abs(fft_shifted))
        # normalization to visualize as image
        vmin, vmax = np.min(magnitude), np.max(magnitude)
        magnitude = (magnitude - vmin) / (vmax - vmin)
        img = Image.fromarray((magnitude * 255).astype(np.uint8))
        self.logger.log_images(img, category, label)

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
        self.log_images(pixels, 'pred', 'Extrapolation', f"{interval}")

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
        captions = [f'{zoom_factor} Reconstruction (Ours)']
        gt_pixels = (gt_pixels * 255).squeeze(-1).numpy().astype(np.uint8)
        cropped = Image.fromarray(gt_pixels).crop(crop_rectangle)
        for filter in self.hyper.get('zoom_filters', ['linear']):
            resized = cropped.resize((w, h), RESIZING_FILTERS[filter])
            images.append(resized)
            captions.append(f"{zoom_factor} Baseline - {filter} interpolation")
             
        self.logger.log_images(images, 'zoom', f"{zoom_factor}x", captions)

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
        
        self.logger.log_images(img, "etc", "Chosen Frequencies")
        # img = wandb.Image(img)
        # wandb.log({"Chosen Frequencies": img})
        
    # def log_detailtensor(self, pixels:torch.Tensor, label:str):
    #     pixels = (pixels + 1.0) / (2.0)
    #     image = wandb.Image(pixels.numpy())    
    #     wandb.log({label: image})


class WandBLogger3D(WandBLogger):

    def on_stage_trained(self, current_model: MRNet,
                                train_loader,
                                test_loader):
        super().on_stage_trained(current_model, train_loader, test_loader)
        device = self.hyper.get('eval_device', 'cpu')
        rng = np.random.default_rng()
        self.w_slice = rng.integers(0, test_loader.size()[-1])
        start_time = time.time()
        
        self.log_data(train_loader, "Train Data")
        gt = self.log_data(test_loader, "Ground Truth", True)   
        pred = self.log_prediction(current_model, test_loader, device)
        self.log_PSNR(gt.cpu(), pred.cpu())
        self.log_SSIM(gt.cpu(), pred.cpu())
        self.log_point_cloud(current_model, device)

        extrapolation_interval = self.hyper.get('extrapolate', None)
        if extrapolation_interval is not None:
            self.log_extrapolation(current_model, 
                                   extrapolation_interval, 
                                    test_loader.shape[1:], device)
        print(f"[Logger] All inference done in {time.time() - start_time}s on {device}")
        current_model.train()
        current_model.to(self.hyper['device'])
        
        if current_model.n_stages() < self.hyper['max_stages']:
            # apparently, we need to finish the run when running on script
            wandb.finish()
       
    def log_data(self, dataset, label, test=False):
        views = self.hyper['slice_views']
        slices = dataset.get_slices(views)
        captions = f"{views}"
        
        join_views = self.hyper.get("join_views", False)

        if (re.match('laplace_*', self.hyper['filter']) 
            and self.hyper['stage'] > 1):
            self.log_detailtensor(slices, label)
        else:
            if join_views:
                slices = torch.vstack((torch.hstack(slices[:2]), 
                                       torch.hstack(slices[2:])))
            else:
                captions = [f"Slice {view} = k" for view in views]

            self.log_images(slices, label, captions)

        if not test:
            return

        if test:
            self.log_fft(slices, f'FFT {label}', views)
            # if self.visualize_gt_grads:

            #     try:
            #         gt_grads = test_loader.sampler.img_grad
            #         self.log_gradmagnitude(gt_grads, 'Ground Truth - Gradient')
            #     except:
            #         print(f'No gradients in sampler and visualization is True. Set visualize_grad to False')
        return slices if join_views else torch.concat(slices)
    
    def log_prediction(self, model, test_loader, device):
        views = self.hyper['slice_views']
        captions = f"{views}"
        dims = test_loader.shape[1:]
        channels = test_loader.shape[0]
        domain_slices = make_domain_slices(dims[0], 
                                           *self.hyper['domain'],
                                           views)
        pred_slices = []
        grads = []
        for slice in domain_slices:
            slice = slice.view(-1, self.hyper['in_features']).to(device)
            output_dict = model(slice)
            model_out = output_dict['model_out']
            grads.append(gradient(model_out, 
                                  output_dict['model_in']).detach().cpu().view(dims[0], dims[1], 3))
            pred_slices.append(
                model_out.view(dims[0], dims[1], channels).detach().cpu())

        join_views = self.hyper.get("join_views", False)
        if join_views:
            pred_slices = torch.vstack(
                (torch.hstack(pred_slices[:2]), torch.hstack(pred_slices[2:])))
        else:
            captions = [f"Slice {view} = k" for view in views]
        
        self.log_images(pred_slices, 'Prediction', captions)
        self.log_fft(pred_slices, 'FFT Prediction', captions)
        self.log_gradmagnitude(grads, 'Prediction - Gradient')
        
        return pred_slices if join_views else torch.concat(pred_slices)
    
    def log_gradmagnitude(self, grads:torch.Tensor, label: str):
        # TODO: add option to not stack
        grads = torch.vstack((torch.hstack(grads[:2]), 
                                    torch.hstack(grads[2:])))
        mag = torch.sqrt(grads[:, :, 0]**2 
                         + grads[:, :, 1]**2 
                         + grads[:, :, 2]**2).numpy()
        gmin, gmax = np.min(mag), np.max(mag)
        img = Image.fromarray(255 * (mag - gmin) / gmax).convert('L')
        wandb.log({f'Gradient Magnitude - {label}': wandb.Image(img)})
    
    def log_fft(self, slices, label:str, captions=None):
        if not isinstance(slices, Sequence):
            slices = [slices]
        if self.hyper['channels'] == 3:
            color_space = self.hyper.get('color_space', 'RGB')
            if color_space == 'RGB':
                slices = [rgb_to_grayscale(s) for s in slices]
            elif color_space == 'YCbCr':
                slices = [s[..., 0] for s in slices]
        imgs = []
        for pixels in slices:
            fourier_tensor = torch.fft.fftshift(
                            torch.fft.fft2(pixels.squeeze(-1)))
            magnitude = np.log(1 + abs(fourier_tensor.numpy()))
            vmin, vmax = np.min(magnitude), np.max(magnitude)
            magnitude = (magnitude - vmin) / (vmax - vmin)
            imgs.append((magnitude * 255).astype(np.uint8))
        if len(imgs) == 1:
            img = Image.fromarray(imgs[0])
        else:
            if self.hyper.get('join_views', False):
                magnitude = np.vstack((np.hstack(imgs[:2]), 
                                       np.hstack(imgs[2:])))
                img = wandb.Image(
                    Image.fromarray(magnitude),
                    caption=captions
                )
            else:
                if captions is None:
                    captions = [None] * len(imgs)
                img = [wandb.Image(Image.fromarray(imgs[i]), 
                                   caption=captions[i]) 
                        for i in range(len(imgs))]

        wandb.log({label: img})


    def log_extrapolation(self, model, interval, dims, device='cpu'):
        w, h, d = dims
        start, end = interval[0], interval[1]
        scale = (end - start) // 2
        neww, newh, newd = int(scale * w), int(scale * h), int(scale * d)

        views = self.hyper['slice_views']
        channels = self.hyper['channels']
        join_views = self.hyper.get("join_views", False)
        captions = f"{views}"
        # TODO different resolutions per axis?
        domain_slices = make_domain_slices(neww, 
                                           start, 
                                           end, 
                                           views)
        pred_slices = []
        for slice in domain_slices:
            with torch.no_grad():
                pred_slices.append(
                    output_on_batched_grid(model, 
                                        slice, 
                                        self.hyper['batch_size'], 
                                        device).view(neww, newh, channels)) 
        if join_views:
            pred_slices = torch.vstack((torch.hstack(pred_slices[:2]), 
                                        torch.hstack(pred_slices[2:])))
        else:
            captions = [f"Slices {view} = k" for view in views]
        self.log_images(pred_slices, 'Extrapolation', captions)

    