import wandb
import torch
import os
import re
import numpy as np

from scipy.fft import fft, fftfreq
from matplotlib import cm
from PIL import Image
from copy import deepcopy
from pathlib import Path
from typing import Sequence, Union
from torch.utils.data import BatchSampler

import warnings
from training.loss import gradient
from datasets.sampler import make_grid_coords
from datasets.utils import make_domain_slices

from .logger import Logger
from datasets.utils import (output_on_batched_dataset, 
                            output_on_batched_grid, rgb_to_grayscale, ycbcr_to_rgb, RESIZING_FILTERS, INVERSE_COLOR_MAPPING)
from networks.mrnet import MRNet, MRFactory
from copy import deepcopy
import time
import trimesh
import skimage
from IPython import embed
from torchvision.transforms.functional import to_tensor, to_pil_image

MODELS_DIR = 'models'
MESHES_DIR = 'meshes'


class WandBLogger(Logger):    

    def on_stage_start(self, current_model, stage_number, updated_hyper=None):
        if updated_hyper:
            for key in updated_hyper:
                self.hyper[key] = updated_hyper[key]
        
        wandb.init(project=self.project, 
                    entity=self.entity, 
                    name=self.runname, 
                    config=self.hyper,
                    settings=self.settings)
        wandb.watch(current_model, log_freq=10, log='all')

    @property
    def runname(self):
        hyper = self.hyper
        stage = f"{hyper['stage']}/{hyper['max_stages']}"
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
        return f"{self.name}_{stage}_{w0}_{hf}_{epochs}_{hl}_{res}_{per}"

    def on_epoch_finish(self, current_model, epochloss):
        log_dict = {f'{key.upper()} loss': value 
                        for key, value in epochloss.items()}
        if len(epochloss) > 1:
            log_dict['Total loss'] = sum(
                            [self.hyper['loss_weights'][k] * epochloss[k] 
                                for k in epochloss.keys()])

        wandb.log(log_dict)

    def on_train_finish(self, trained_model, total_epochs):
        save_format = self.hyper.get('save_format', None)
        if save_format is not None:
            self.log_model(trained_model, save_format)
        
        wandb.finish()
        print(f'Total model parameters = ', trained_model.total_parameters())
        print(f'Training finished after {total_epochs} epochs')

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

    def log_PSNR(self, gt, pred):
        clamped = pred.clamp(0, 1)
        mse = torch.mean((gt - clamped)**2)
        psnr = 10 * torch.log10(1 / mse)

        # sanity check
        transform = INVERSE_COLOR_MAPPING[self.hyper.get('color_space', 'RGB')]
        int_gt = (transform(gt) * 255).numpy().astype(np.uint8)
        int_pred = (transform(clamped).clamp(0, 1) * 255
                                ).numpy().astype(np.uint8)
        ski_psnr = skimage.metrics.peak_signal_noise_ratio(int_gt, 
                                                           int_pred,
                                                           data_range=255)
        print(ski_psnr)
        
        label = f"Stage {self.hyper['stage']}"
        table = wandb.Table(data=[(label, psnr, ski_psnr)], 
                            columns = ["Stage", "PSNR", "uint8_PSNR"])
        wandb.log({"psnr_value" : wandb.plot.bar(table, "Stage", "PSNR",
                                    title="PSNR for reconstruction")})
        

    def log_SSIM(self, gt, pred):
        clamped = pred.clamp(0, 1)
        transform = INVERSE_COLOR_MAPPING[self.hyper.get('color_space', 'RGB')]
        ssim = skimage.metrics.structural_similarity(
                        (transform(gt).cpu().numpy() * 255).astype(np.uint8), 
                        (transform(clamped).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8),
                        data_range=1, channel_axis=-1)
        label = f"Stage {self.hyper['stage']}"
        table = wandb.Table(data=[(label, ssim)], columns = ["Stage", "SSIM"])
        wandb.log({"ssim_value" : wandb.plot.bar(table, "Stage", "SSIM",
                                    title="SSIM for reconstruction")})
        
    def log_images(self, pixels, label, captions=None):
        if isinstance(pixels, torch.Tensor):
            pixels = [pixels]
        if captions is None:
            captions = [None] * len(pixels)
        if isinstance(captions, str):
            captions = [captions]
        if len(pixels) != len(captions):
            raise ValueError("label and pixels should have the same size")
        
        color_transform = INVERSE_COLOR_MAPPING[self.hyper.get(
                                                'color_space', 'RGB')]
        pixels = [color_transform(p.cpu()).clamp(0, 1) 
                  for p in pixels]
        
        images = [wandb.Image(pixels[i].squeeze(-1).numpy(),
                              caption=captions[i]) for i in range(len(pixels))]
        wandb.log({label: images})

    def log_detailtensor(self, pixels:torch.Tensor, label:str):
        pixels = (pixels + 1.0) / (2.0)
        image = wandb.Image(pixels.numpy())    
        wandb.log({label: image})


class WandBLogger1D(WandBLogger):

    def on_stage_trained(self, current_model: MRNet,
                                train_loader, test_loader):
        super().on_stage_trained(current_model, train_loader, test_loader)

        device = self.hyper.get('eval_device', 'cpu')
        testX = test_loader.sampler.coords 
        testY = test_loader.data.view(-1, self.hyper['channels'])
        # we'll need samples sorted for FFT
        # xtensor, sortidx = torch.sort(testX)
        if isinstance(testY, dict):
            testY = testY['d0']

        with torch.no_grad():
            fulloutput = current_model(testX.to(device))

        pred = fulloutput['model_out']
        
        self.log_fft(testY, pred)

        self.log_results(current_model, train_loader, test_loader, device)
        
        current_model.train()
        

    def get_fft(self, data):
        W = data.view(-1).cpu().detach().numpy()
        N = len(W)
        yf = fft(W)
        xf = fftfreq(N, 2/N)[:N//2]
        return [[x,y] for (x,y) in zip(xf, 2.0/N * np.abs(yf[0:N//2]) )]

    def log_fft(self, testY, pred):
        ##FFT plot
        fftdata = self.get_fft(testY)
        fft_table = wandb.Table(data = fftdata, columns=["frequency", "FFT"])
        wandb.log({"fft_input": wandb.plot.line(fft_table, "frequency", "FFT", stroke=None, title="Fast Fourier Transform")})

        predfftdata = self.get_fft(pred)
        pred_fft_table = wandb.Table(data = predfftdata, 
                                    columns=["frequency", "FFT"])

        wandb.log({"FFT_approximation" : wandb.plot.line_series(
                            xs=[fft_table.get_column("frequency"),
                                pred_fft_table.get_column("frequency")],
                            ys=[fft_table.get_column("FFT"), 
                                pred_fft_table.get_column("FFT")],
                            keys=["Ground Truth", "Prediction"],
                            title="FFT Approximation",
                            xname="frequency",)
        })

    def log_results(self, model, train_loader, test_loader, device):
        trainX = train_loader.sampler.coords 
        trainY = train_loader.data.view(-1, self.hyper['channels'])#next(iter(train_loader))
        testX = test_loader.sampler.coords 
        testY = test_loader.data.view(-1, self.hyper['channels'])#next(iter(test_loader))

        if not isinstance(testY, dict):
            testY = {'d0': testY}
        if not isinstance(trainY, dict):
            trainY = {'d0': trainY}
        
        X_train = trainX.view(-1).detach().numpy()
        X_test = testX.view(-1).detach().numpy()
        Y_train = trainY['d0'].view(-1).detach().numpy()
        Y_test = testY['d0'].view(-1).detach().numpy()

        self.log_ground_truth_signal(X_train, Y_train, X_test, Y_test)
        # Prediction
        model.eval()
        
        tst = trainX.to(device)
        output_dict = model(tst)
        
        train_pred = output_dict['model_out']
        output_dict = model(testX.to(device))
        test_pred, coords = output_dict['model_out'], output_dict['model_in']

        # Log prediction curve
        pred_table = wandb.Table(
                        data=[[x,y] for (x,y) in zip(X_test, test_pred.cpu().view(-1).detach().numpy())],
                        columns=["time","amplitude"])
        wandb.log({"pred_only": wandb.plot.line(pred_table, "time","amplitude", stroke=None, title="Prediction")})
        
        # Log results for training samples
        self.log_regression_curves(X_train,
                                    Y_train,
                                    train_pred.cpu().view(-1).detach().numpy(),
                                    'train_plot',
                                    "Regression - TRAIN samples")
        
        # Log results for test samples
        self.log_regression_curves(X_test,
                                    Y_test,
                                    test_pred.cpu().view(-1).detach().numpy(),
                                    'pred_plot',
                                    "Regression - TEST samples")
        
        gt_d1 = testY.get('d1', None)
        gt_d2 = testY.get('d2', None)
        if gt_d1 is not None:
            model_d1 = gradient(test_pred, coords)
            self.log_regression_curves(X_test, 
                                gt_d1.view(-1).detach().numpy(),
                                model_d1.cpu().view(-1).detach().numpy(),
                                'd1_plot',
                                "Regression of Gradient - test samples")
        if gt_d2 is not None:
            model_d2 = gradient(model_d1, coords)
            self.log_regression_curves(X_test, 
                        gt_d2.view(-1).detach().numpy(),
                        model_d2.cpu().view(-1).detach().numpy(),
                        'd2_plot',
                        "Regression of D2 - test samples")
        
        extrapolation_interval = self.hyper.get('extrapolate', None)
        if extrapolation_interval is not None:
            self.log_extrapolation(model, extrapolation_interval,
                                    X_test, Y_test, device)
            
        self.log_chosen_frequencies(model, None)

        
    def log_regression_curves(self, X, gt, pred, plot_name, title):
        wandb.log(
            {plot_name: wandb.plot.line_series(
                xs = X,
                ys = [gt, pred],
                keys = ["Ground Truth", "Prediction"],
                title = title,
                xname = "time"
            )}
        )

    def log_ground_truth_signal(self, X_train, Y_train, X_test, Y_test):
        wandb.log(
                {'gt_plot': wandb.plot.line_series(
                    xs = [X_train, X_test],
                    ys = [Y_train, Y_test],
                    keys = ["TRAIN data", "TEST data"],
                    title = "Ground Truth Signal",
                    xname = "time"
                )}
        )

    def log_extrapolation(self, model, interval, X_test, Y_test, device):
        space = X_test[1] - X_test[0]
        start, end = interval[0], interval[1]
        newsamplesize = abs(int((end - start) / space)) + 1
        ext_x = torch.linspace(start, end, newsamplesize)
        out_dict = model(ext_x.view(-1, 1).to(device))
        ext_y = out_dict['model_out'].cpu().view(-1).detach()
        wandb.log(
                {'extrapolation_plot': wandb.plot.line_series(
                    xs = [X_test, ext_x.numpy()],
                    ys = [Y_test, ext_y.numpy()],
                    keys = ["TEST data", "Prediction"],
                    title = "Extrapolation Prediction",
                    xname = "time"
                )}
        )

    def log_chosen_frequencies(self, model, predfftdata):
        nyquist_limit = self.hyper['width']//4
        all_freqs = np.arange(-nyquist_limit, nyquist_limit+1)
        frequencies = []
        for stage in model.stages:
            last_stage_frequencies = stage.first_layer.linear.weight.cpu()
            freqs = np.round(self.hyper['period'] 
                    * last_stage_frequencies.view(-1).numpy() 
                    / (2 * np.pi))
            frequencies.append(freqs)
        frequencies = np.concatenate(frequencies)
        # print(frequencies.shape, frequencies)
        values = np.zeros_like(all_freqs)
        values[np.in1d(all_freqs, frequencies.astype(np.int32)).nonzero()] = 1
        wandb.log(
            {"init_freqs": wandb.plot.line_series(
                xs = all_freqs,
                ys = [values],
                keys = ["chosen"],
                title = "chosen frequencies",
                xname = "freqs"
            )}
        )
        print("Logged frequencies")
        ##FFT plot
        # predfftdata = self.get_fft(pred)
        
        # freq_table = wandb.Table(data = np.stack(frequencies, 
        #                                          np.ones_like(frequencies) * 0.05), 
        #                             columns=["frequency", "FFT"])
        # wandb.log({"frequencies": wandb.plot.line(freq_table, "freq","chosen", stroke=None, title="Chosen Frequencies")})


class WandBLogger2D(WandBLogger):

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
        self.log_PSNR(gt.cpu(), pred.cpu())
        self.log_SSIM(gt.cpu(), pred.cpu())

        extrapolation_interval = self.hyper.get('extrapolate', None)
        if extrapolation_interval is not None:
            self.log_extrapolation(current_model, extrapolation_interval, 
                                    test_loader.size()[1:], device)
        zoom = self.hyper.get('zoom', [])
        for zfactor in zoom:
            self.log_zoom(current_model, test_loader, zfactor, device)

        # if current_model.period > 0:
        #     self.log_chosen_frequencies(current_model)
        
        print(f"[Logger] All inference done in {time.time() - start_time}s on {device}")
        current_model.train()
        current_model.to(self.hyper['device'])
        
        if current_model.n_stages() < self.hyper['max_stages']:
            # apparently, we need to finish the run when running on script
            wandb.finish()
       
    def log_traindata(self, train_loader):
        pixels = train_loader.data.permute((1, 2, 0))
        #pixels = self.as_imagetensor(torch.clamp(traindata, 0, 1))
        if re.match('laplace_*', self.hyper['filter']) and self.hyper['stage'] > 1:
            self.log_detailtensor(pixels, 'Train Data')
        else:
            # try:
            #     color_transform = INVERSE_COLOR_MAPPING[
            #                 self.hyper.get('color_space', 'RGB')]
            #     pixels = color_transform(pixels.cpu()).clamp(0, 1)
            #     print("AQUI????")
            #     img = wandb.Image(pixels.squeeze(-1).numpy(), 
            #                 masks={"mask_data": train_loader.domain_mask})
            #     wandb.log({"Train Data": img})
            # except Exception as e:
            if train_loader.domain_mask is not None:
                # print(e, "???")
                # pixels = pixels.cpu()
                mask = train_loader.domain_mask.float().unsqueeze(-1)
                # print(mask.shape, "mask")
                # avg = pixels.mean(dim=-1).unsqueeze(-1)
                # print(avg.shape, "avg")
                #pixels[mask] = avg[mask]
                pixels = pixels * mask
                values = [pixels]
                captions = [f"{list(train_loader.shape)}"]
            else: 
                values = [pixels]
                captions = [f"{list(train_loader.shape)}"]
            self.log_images(values, 'Train Data', captions)
    
    def log_groundtruth(self, dataset):
        # permute to H x W x C
        pixels = dataset.data.permute((1, 2, 0))
        if re.match('laplace_*', self.hyper['filter']) and self.hyper['stage'] > 1:
            self.log_detailtensor(pixels, 'Ground Truth')
        else:
            self.log_images([pixels], 
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
        
        self.log_fft(gray_pixels, 'FFT Ground Truth')

        if 'd1' in self.hyper['attributes']:
            grads = dataset.data_attributes['d1']
            
            if color_space == 'YCbCr':
                    grads = grads[0, ...]
            elif color_space == 'RGB':
                grads = (0.2126 * grads[0, ...] 
                        + 0.7152 * grads[1, ...] 
                        + 0.0722 * grads[2, ...])
            elif color_space == 'L':
                grads = grads.squeeze(0)

            mag = np.hypot(grads[:, :, 0].squeeze(-1).numpy(),
                           grads[:, :, 1].squeeze(-1).numpy())
            vmin, vmax = np.min(mag), np.max(mag)
            img = Image.fromarray(255 * (mag - vmin) / (vmax - vmin)).convert('L')
            wandb.log({f'Gradient Magnitude - {"GT"}': wandb.Image(img)})
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

    def log_detailtensor(self, pixels:torch.Tensor, label:str):
        pixels = (pixels + 1.0) / (2.0)
        image = wandb.Image(pixels.numpy())    
        wandb.log({label: image})
    
    def log_gradmagnitude(self, grads:torch.Tensor, label: str):
        mag = np.hypot(grads[:, :, 0].squeeze(-1).numpy(),
                        grads[:, :, 1].squeeze(-1).numpy())
        gmin, gmax = np.min(mag), np.max(mag)
        img = Image.fromarray(255 * (mag - gmin) / (gmax - gmin)).convert('L')
        wandb.log({f'Gradient Magnitude - {label}': wandb.Image(img)})
    
    def log_fft(self, pixels:torch.Tensor, label:str, captions=None):
        '''Assumes a grayscale version of the image'''
        
        fft_pixels = torch.fft.fft2(pixels)
        fft_shifted = torch.fft.fftshift(fft_pixels).numpy()
        magnitude =  np.log(1 + abs(fft_shifted))
        # normalization to visualize as image
        vmin, vmax = np.min(magnitude), np.max(magnitude)
        magnitude = (magnitude - vmin) / (vmax - vmin)
        img = Image.fromarray((magnitude * 255).astype(np.uint8))
        wandb.log({label: wandb.Image(img)})

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
        self.log_images(pixels, 'Extrapolation', f"{interval}")
        # norm_weights = []
        # if (self.hyper['channels'] == 1 
        #     and self.hyper['loss_weights']['d0'] == 0):
        #     norm_weights = [0, 1]
        # self.log_imagetensor(pixels, 'Extrapolation', norm_weights)

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
        pixels = color_transform(pixels)
        gt_pixels = color_transform(gt_pixels)

        pixels = (pixels.clamp(0, 1) * 255).squeeze(-1).numpy().astype(np.uint8)
        images = [
            wandb.Image(Image.fromarray(pixels), 
                        caption='Reconstruction (Ours)')
        ]
        gt_pixels = (gt_pixels * 255).squeeze(-1).numpy().astype(np.uint8)
        cropped = Image.fromarray(gt_pixels).crop(crop_rectangle)
        for filter in self.hyper.get('zoom_filters', ['linear']):
            resized = cropped.resize((w, h), RESIZING_FILTERS[filter])
            images.append(
                wandb.Image(resized, 
                            caption=f"Baseline - {filter} interpolation")
            )
        wandb.log({f"Zoom {zoom_factor}x": images})

    def log_chosen_frequencies(self, model: MRNet):
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
        img = wandb.Image(img)
        wandb.log({"Chosen Frequencies": img})
        


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