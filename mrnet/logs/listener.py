from IPython import embed
import numpy as np
import os
import skimage
import time
import torch
from PIL import Image
from pathlib import Path
from scipy.fft import fft, fftfreq
from torch.utils.data import BatchSampler
from mrnet.datasets.signals import Signal1D
from mrnet.training.loss import gradient
from mrnet.datasets.sampler import make_grid_coords
from mrnet.datasets.utils import (output_on_batched_dataset, 
                            output_on_batched_grid, rgb_to_grayscale, ycbcr_to_rgb, RESIZING_FILTERS, INVERSE_COLOR_MAPPING)
from mrnet.networks.mrnet import MRNet, MRFactory
from mrnet.logs.logger import LocalLogger, Logger, WandBLogger


MODELS_DIR = 'models'
MESHES_DIR = 'meshes'


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

        self.handler = ResultHandler.from_dict(hyper)

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
        logger = LoggerClass(self.project,
                                    self.name,
                                    self.hyper,
                                    self.basedir,
                                    stage=stage_number, 
                                    entity=self.entity, 
                                    config=self.config, 
                                    settings=self.settings)
        logger.prepare(current_model)
        self.handler.logger = logger

    def on_stage_trained(self, current_model: MRNet,
                                train_loader,
                                test_loader):
        device = self.hyper.get('eval_device', 'cpu')
        current_stage = current_model.n_stages()
        current_model.eval()
        current_model.to(device)
        
        start_time = time.time()
        
        self.handler.log_chosen_frequencies(current_model)
        gt = self.handler.log_groundtruth(test_loader,
                                          train_loader,
                                          stage=current_stage)
        pred = self.handler.log_prediction(current_model, test_loader, device)
        self.handler.log_metrics(gt.cpu(), pred.cpu())
        self.handler.log_extrapolation(current_model,
                                       test_loader,
                                       device)
        self.handler.log_zoom(current_model, test_loader, device)
        # TODO: check for pointcloud data
        print(f"[Logger] All inference done in {time.time() - start_time}s on {device}")
        current_model.train()
        current_model.to(self.hyper['device'])
        self.handler.log_model(current_model)

    def on_batch_finish(self, batchloss):
        pass

    def on_epoch_finish(self, current_model, epochloss):
        self.handler.log_losses(epochloss)

    def on_train_finish(self, trained_model, total_epochs):
        self.handler.finish()
        print(f'Training finished after {total_epochs} epochs')


class ResultHandler:

    def from_dict(hyper):
        dim = hyper['in_features']
        HANDLERS = {
            1: Signal1DHandler,
            2: ImageHandler
        }
        return HANDLERS[dim](hyper)
        

    def __init__(self, hyper, logger=None) -> None:
        self.hyper = hyper
        self.logger: Logger = logger

    def log_losses(self, epochloss):
        log_dict = {f'{key.upper()} loss': value 
                        for key, value in epochloss.items()}
        if len(epochloss) > 1:
            log_dict['Total loss'] = sum(
                            [self.hyper['loss_weights'][k] * epochloss[k] 
                                for k in epochloss.keys()])
        self.logger.log_losses(log_dict)

    def log_metrics(self, gt, pred):
        self.log_PSNR(gt, pred)

    def log_SSIM(self, gt, pred):
        #clamped = pred.clamp(0, 1)
        transform = INVERSE_COLOR_MAPPING[self.hyper.get('color_space', 'RGB')]
        ssim = skimage.metrics.structural_similarity(
                        (transform(gt).cpu().numpy() * 255).astype(np.uint8), 
                        (transform(pred).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8),
                        data_range=1, channel_axis=-1)
        label = f"Stage {self.hyper['stage']}"
        self.logger.log_metric("ssim", ssim, label)

    def log_PSNR(self, gt, pred):
        mse = torch.mean((gt - pred)**2)
        psnr = 10 * torch.log10(1 / mse)
        
        label = f"Stage {self.hyper['stage']}"
        self.logger.log_metric("psnr", psnr, label)

    def log_model(self, model:MRNet):
        print(f'Total model parameters = ', model.total_parameters())
        name = f"{model.class_code()}_stg{model.n_stages()}"
        self.logger.log_model(model, path='tmp', fname=name)

    def finish(self):
        self.logger.finish()

    def log_groundtruth(self, test_loader, train_loader, **kwargs):
        raise NotImplementedError

    def log_prediction(self, model, test_loader, device):
        raise NotImplementedError

    def log_gradmagnitude(self, grads:torch.Tensor, label: str, **kw):
        pass

    def log_fft(self, pixels:torch.Tensor, label:str, 
                    captions=None, **kw):
        raise NotImplementedError
    
    def log_chosen_frequencies(self, model: MRNet):
        if model.period <= 0:
            return
    
    def log_zoom(self, model, test_loader, device):
        raise NotImplementedError
    
    def log_extrapolation(self, model, test_loader, device='cpu'):
        raise NotImplementedError

    
class ImageHandler(ResultHandler):

    def log_metrics(self, gt, pred):
        super().log_metrics(gt, pred)
        self.log_SSIM(gt, pred)

    def log_traindata(self, train_loader, **kw):
        pixels = train_loader.data.permute((1, 2, 0))
        print(pixels.shape, "DEBUG 1")
        if train_loader.domain_mask is not None:
            mask = train_loader.domain_mask.float().unsqueeze(-1)
            pixels = pixels * mask
        values = [pixels]
        captions = [f"{list(train_loader.shape)}"]
        
        self.logger.log_images(values, 'Train Data', captions, category='gt')

    def log_groundtruth(self, test_loader, train_loader, **kwargs):
        self.log_traindata(train_loader, **kwargs)
        # permute to H x W x C
        pixels = test_loader.data.permute((1, 2, 0))
        
        self.logger.log_images([pixels], 
                            'Ground Truth', 
                            [f"{list(test_loader.shape)}"],
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
            grads = test_loader.data_attributes['d1']
            
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
            
        return test_loader.data.permute((1, 2, 0)
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

    def log_chosen_frequencies(self, model: MRNet):
        super().log_chosen_frequencies(model)
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

    def log_extrapolation(self, model, test_loader, device='cpu'):
        try:
            interval = self.hyper['extrapolate']
        except KeyError:
            return
        w, h = test_loader.size()[1:]
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

    def log_zoom(self, model, test_loader, device):
        w, h = test_loader.shape[1:]
        domain = self.hyper.get('domain', [-1, 1])
        zoom = self.hyper.get('zoom', [])
        for zoom_factor in zoom:
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
            
class Signal1DHandler(ResultHandler):
    def log_traindata(self, dataset: Signal1D, **kwargs):
        X = dataset.coords.view(-1).numpy()
        Y = dataset.data.view(-1).numpy()

        if dataset.domain_mask is not None:
            raise NotImplementedError()
        
        captions = [f"{list(dataset.shape)}"]
        self.logger.log_graph([X], [Y], 'Train Data',
                              captions=captions,
                              category='gt')
        
    def log_groundtruth(self, test_loader, train_loader, **kwargs):
        self.log_traindata(train_loader, **kwargs)
        trainX = train_loader.coords.view(-1).numpy()
        trainY = train_loader.data.view(-1).numpy()
        testX = test_loader.coords.view(-1).numpy()
        testY = test_loader.data.view(-1).numpy()
        captions = [f"Train samples", "Test signal"]
        
        self.logger.log_graph([trainX, testX], [trainY, testY], 
                              'Ground Truth',
                              captions=captions,
                              category='gt',
                              fname='ground_truth',
                              **kwargs)

        self.log_fft({'train': trainY, 'test': testY}, 
                     captions=captions,
                     category='gt',
                     label="FFT Test vs Train samples", 
                     fname='fft_test_train',
                     **kwargs)
        
        return torch.from_numpy(testY)
    
    def log_prediction(self, model, test_loader, device):
        X = test_loader.coords
        testY = test_loader.data.view(-1).numpy()

        output_dict = model(X.to(device))
        pred = output_dict['model_out'].detach().cpu().view(-1).numpy()
        captions = ['Test signal', 'Prediction']
        self.logger.log_graph(X.view(-1).numpy(),
                              [testY, pred],
                              "Prediction vs Test signal",
                              captions=captions,
                              category='pred',
                              fname='pred_test')
        
        self.log_fft({'test': testY, 'pred': pred},
                     label="FFT Prediction vs Test signal",
                     captions=captions,
                     category='pred',
                     fname='fft_pred_test')
        return torch.from_numpy(pred)
        
    def log_extrapolation(self, model, test_loader, device='cpu'):
        try:
            interval = self.hyper['extrapolate']
        except KeyError:
            return
        X = test_loader.coords.view(-1)
        Y = test_loader.data.view(-1)
        start, end = interval[0], interval[1]
        scale = ((end - start) / 
                 (test_loader.domain[1] - test_loader.domain[0]))
        newsamplesize = int(len(X) * scale)
        
        ext_x = torch.linspace(start, end, newsamplesize)
        out_dict = model(ext_x.view(-1, 1).to(device))
        ext_y = out_dict['model_out'].cpu().view(-1).detach()

        captions = ["TEST data", "Prediction"]
        self.logger.log_graph([X.numpy(), ext_x.numpy()], 
                              [Y.numpy(), ext_y.numpy()],
                              fname='extrapolation',
                              label="Extrapolation Prediction",
                              captions=captions,
                              category='pred')

    def log_fft(self, data, **kwargs):
        ##FFT plot
        Xs, Ys = [], []
        for key, value in data.items():
            N = len(value)
            Xs.append(fftfreq(N, 2/N)[:N//2])
            yf = fft(value)
            Ys.append(2.0 / N * np.abs(yf[0:N//2]))
            
        self.logger.log_graph(Xs, Ys, **kwargs)

    def log_zoom(self, model, test_loader, device):
        nsamples = len(test_loader.coords)
        domain = self.hyper.get('domain', [-1, 1])
        zoom = self.hyper.get('zoom', [])
        for zoom_factor in zoom:
            start, end = domain[0]/zoom_factor, domain[1]/zoom_factor
            zoom_coords = torch.linspace(start, end, nsamples).view(-1, 1)
            with torch.no_grad():
                output_dict = model(zoom_coords.to(device))
                values = output_dict['model_out'].cpu().view(-1)
            
            captions = ["Model - refined grid", "Naive - old grid"]
            c1 = nsamples * (zoom_factor - 1) // (2 * zoom_factor)
            c2 = nsamples * (zoom_factor + 1) // (2 * zoom_factor)
            X = test_loader.coords.view(-1)[c1:c2]
            Y = test_loader.data.view(-1)[c1:c2]
            self.logger.log_graph([zoom_coords, X],
                                  [values, Y],
                                  f"{zoom_factor}x Zoom - Model vs Naive",
                                  category='zoom',
                                  fname=f'zoom_{zoom_factor}x',
                                  captions=captions)


