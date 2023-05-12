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
from typing import Sequence
from torch.utils.data import BatchSampler

import warnings
from training.loss import gradient
from datasets.sampler import make_grid_coords

from .logger import Logger
from .utils import output_per_batch, ycbcr_to_rgb
from networks.mrnet import MRNet, MRFactory
from copy import deepcopy
import time
import trimesh
import skimage
from IPython import embed
from torchvision.transforms.functional import to_tensor

MODELS_DIR = 'models'
MESHES_DIR = 'meshes'


class WandBLogger(Logger):
    def __init__(self, project: str, 
                        name: str, 
                        hyper: dict,
                        basedir: str, 
                        entity=None, 
                        config=None, 
                        settings=None):
        super().__init__(project, name, hyper, 
                         basedir, entity, config, settings)

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
        psnr = 10*torch.log10(1 / (torch.mean(gt - pred)**2 + 1e-10))
        
        label = f"Stage {self.hyper['stage']}"
        table = wandb.Table(data=[(label, psnr)], columns = ["Stage", "PSNR"])
        wandb.log({"psnr_value" : wandb.plot.bar(table, "Stage", "PSNR",
                                    title="PSNR for reconstruction")})

    def log_SSIM(self, gt, pred):
        ssim = skimage.metrics.structural_similarity(gt.detach().cpu().numpy(), 
                                                    pred.detach().cpu().numpy(),
                                                    data_range=1, channel_axis=-1)
        label = f"Stage {self.hyper['stage']}"
        table = wandb.Table(data=[(label, ssim)], columns = ["Stage", "SSIM"])
        wandb.log({"ssim_value" : wandb.plot.bar(table, "Stage", "SSIM",
                                    title="SSIM for reconstruction")})



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
        print(frequencies.shape, frequencies)
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
        self.log_traindata(train_loader)
        gt = self.log_groundtruth(test_loader)
        pred = self.log_prediction(current_model, test_loader, device)
        self.log_PSNR(gt.to(device), pred.to(device))
        self.log_SSIM(gt.cpu(), pred.cpu())

        extrapolation_interval = self.hyper.get('extrapolate', None)
        if extrapolation_interval is not None:
            self.log_extrapolation(current_model, extrapolation_interval, 
                                    test_loader.size()[1:], device)
        zoom = self.hyper.get('zoom', [])
        for zfactor in zoom:
            self.log_zoom(current_model, test_loader, zfactor, device)
        
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
            self.log_imagetensor(pixels, 'Train Data')
    
    def log_groundtruth(self, test_loader):
        gtdata = test_loader.data.view(-1, self.hyper['channels'])
        #pixels = self.as_imagetensor(torch.clamp(gtdata, 0, 1))
        pixels = test_loader.data.permute((1, 2, 0))
        if re.match('laplace_*', self.hyper['filter']) and self.hyper['stage'] > 1:
            self.log_detailtensor(pixels, 'Ground Truth')
        else:
            self.log_imagetensor(pixels, 'Ground Truth')
        
        self.log_fft(pixels, 'FFT Ground Truth')

        if 'd1' in self.hyper['attributes']:
            grads = test_loader.data_attributes['d1']
            if self.hyper['channels'] == 3:
                if self.hyper['YCbCr']:
                    grads = grads[0, ...]
                else:
                    grads = (0.2126 * grads[0, ...] 
                        + 0.7152 * grads[1, ...] 
                        + 0.0722 * grads[2, ...])
            else:
                grads = grads.squeeze(0)
            mag = np.hypot(grads[:, :, 0].squeeze(-1).numpy(),
                    grads[:, :, 1].squeeze(-1).numpy())
            gmin, gmax = np.min(mag), np.max(mag)
            img = Image.fromarray(255 * (mag - gmin) / (gmax - gmin)).convert('L')
            wandb.log({f'Gradient Magnitude - {"GT"}': wandb.Image(img)})
        return gtdata
    
    def log_prediction(self, model, test_loader, device):
        coords = test_loader.sampler.coords
        pixels = []
        grads = []
        for batch in BatchSampler(coords, 
                                  self.hyper['batch_size'], 
                                  drop_last=False):
            batch = torch.stack(batch)
            output_dict = model(batch.to(device))
            pixels.append(output_dict['model_out'].detach().cpu())
            value = output_dict['model_out']
            if self.hyper['channels'] == 3:
                if self.hyper.get('YCbCr', False):
                    value = value[:, 0:1]
                else:
                    value = (0.2126 * value[:, 0:1] 
                        + 0.7152 * value[:, 1:2] 
                        + 0.0722 * value[:, 2:3])
            grads.append(gradient(value, 
                                  output_dict['model_in']).detach().cpu())
        pixels = torch.concat(pixels)
        grads = torch.concat(grads)
        h, w = test_loader.size()[1:]
        pred_pixels = pixels.reshape((h, w, self.hyper['channels']))
        norm_weights = []
        if (self.hyper['channels'] == 1 
            and self.hyper['loss_weights']['d0'] == 0):
            norm_weights = [torch.min(test_loader.data), torch.max(test_loader.data)]
        self.log_imagetensor(pred_pixels, 'Prediction', norm_weights)
        self.log_fft(pred_pixels, 'FFT Prediction')

        # model_grads = gradient(model_out, output_dict['model_in'])
        grads = grads.reshape((h, w, 2))
        self.log_gradmagnitude(grads, 'Prediction - Gradient')
        return pixels

    def log_imagetensor(self, pixels:torch.Tensor, label:str, norm_weights=[]):
        if norm_weights:
            vmin, vmax = norm_weights
            pmin, pmax = torch.min(pixels), torch.max(pixels)
            pixels = (pixels - pmin) / (pmax - pmin)
            pixels = pixels * (vmax - vmin) + vmin

        if self.hyper.get('YCbCr', False) and self.hyper['channels'] == 3:
            pixels = ycbcr_to_rgb(pixels)
        image = wandb.Image(pixels.clamp(0, 1).numpy())
        wandb.log({label: image})

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
    
    def log_fft(self, pixels:torch.Tensor, label:str):
        if self.hyper['channels'] == 3:
            if self.hyper.get('YCbCr', False):
                pixels = pixels[..., 0]
            else:
                pixels = (0.2126 * pixels[:, :, 0] 
                        + 0.7152 * pixels[:, :, 1] 
                        + 0.0722 * pixels[:, :, 2])
        fourier_tensor = torch.fft.fftshift(
                        torch.fft.fft2(pixels.squeeze(-1)))
        magnitude = 20 * np.log(abs(fourier_tensor.numpy()) + 1e-10)
        magnitude = magnitude / np.max(magnitude)
        graymap = cm.get_cmap('gray')
        img = Image.fromarray(np.uint8(graymap(magnitude) * 255))
        wandb.log({label: wandb.Image(img)})

    def log_extrapolation(self, model, interval, dims, device='cpu'):
        w, h = dims
        start, end = interval[0], interval[1]
        scale = (end - start) // 2
        neww, newh = int(scale * w), int(scale * h)
        
        ext_domain = make_grid_coords((neww, newh), start, end, dim=2)
        with torch.no_grad():
            pixels = []
            for batch in BatchSampler(ext_domain, 
                                      self.hyper['batch_size'], 
                                      drop_last=False):
                batch = torch.stack(batch)
                output_dict = model(batch.to(device))
                pixels.append(output_dict['model_out'].detach().cpu().clamp(0, 1))
            pixels = torch.concat(pixels)

        pixels = pixels.view((newh, neww, self.hyper['channels']))
        norm_weights = []
        if (self.hyper['channels'] == 1 
            and self.hyper['loss_weights']['d0'] == 0):
            norm_weights = [0, 1]
        self.log_imagetensor(pixels, 'Extrapolation', norm_weights)

    def log_zoom(self, model, test_loader, zoom_factor, device):
        w, h = test_loader.size()[1:]
        domain = self.hyper.get('domain', [-1, 1])
        start, end = domain[0]/zoom_factor, domain[1]/zoom_factor
        zoom_coords = make_grid_coords((w, h), start, end, dim=2)
        with torch.no_grad():
            pixels = []
            for batch in BatchSampler(zoom_coords, 
                                      self.hyper['batch_size'], 
                                      drop_last=False):
                batch = torch.stack(batch)
                output_dict = model(batch.to(device))
                pixels.append(output_dict['model_out'].detach().cpu())
            pixels = torch.concat(pixels)
        # center crop
        cropsize = int(w // zoom_factor)
        left, top = (w - cropsize), (h - cropsize)
        right, bottom = (w + cropsize), (h + cropsize)
        crop_rectangle = tuple(np.array([left, top, right, bottom]) // 2)
        gt = Image.fromarray(
                            (test_loader.data.permute((1, 2, 0)
                              ).squeeze(-1).numpy() * 255).astype(np.uint8)
            ).crop(crop_rectangle).resize((w, h), Image.Resampling.BICUBIC)
        
        pixels = pixels.view((h, w, self.hyper['channels']))
        if (self.hyper['channels'] == 1 
            and self.hyper['loss_weights']['d0'] == 0):
            vmin = torch.min(test_loader.data)
            vmax = torch.max(test_loader.data)
            pmin, pmax = torch.min(pixels), torch.max(pixels)
            pixels = (pixels - pmin) / (pmax - pmin)
            pixels = pixels * vmax #(vmax - vmin) + vmin
        # images = [wandb.Image(pixels.clamp(0, 1).numpy(), caption='Pred'),
        #           wandb.Image(gt, caption='GT (bicubic)')
        #           ]
        if self.hyper.get('YCbCr', False) and self.hyper['channels'] == 3:
            pixels = ycbcr_to_rgb(pixels)

        # print(pixels.shape, gt.shape)
        imgs = torch.hstack([pixels.clamp(0, 1), 
                             to_tensor(gt).permute((1, 2, 0))])
        imgs = wandb.Image(imgs.squeeze(-1),
                            caption="Pred (left); GT - bicubic (right)")
        wandb.log({f"Zoom {zoom_factor}x": imgs})


class WandBLogger3D(WandBLogger):

    def __init__(self, project: str, name: str, hyper: dict, basedir: str, entity=None, config=None, settings=None):
        super().__init__(project, name, hyper, basedir, entity, config, settings)
        self.x_slice = 1
        self.y_slice = 2
        self.z_slice = 3
        self.w_slice = 1


    def on_stage_trained(self, current_model: MRNet,
                                train_loader,
                                test_loader):
        super().on_stage_trained(current_model, train_loader, test_loader)
        device = self.hyper.get('eval_device', 'cpu')
        rng = np.random.default_rng()
        self.w_slice = rng.integers(0, test_loader.size()[-1])
        start_time = time.time()
        
        self.log_traindata(train_loader)
        gt = self.log_groundtruth(test_loader)   
        pred = self.log_prediction(current_model, test_loader, device)
        self.log_PSNR(gt.to(device), pred)
        self.log_SSIM(gt.cpu(), pred.cpu())
        self.log_point_cloud(current_model, device)

        extrapolation_interval = self.hyper.get('extrapolate', None)
        if extrapolation_interval is not None:
            self.log_extrapolation(current_model, extrapolation_interval, 
                                    test_loader.size()[1:], device)
        print(f"[Logger] All inference done in {time.time() - start_time}s on {device}")
        current_model.train()
        current_model.to(self.hyper['device'])
        
        if current_model.n_stages() < self.hyper['max_stages']:
            # apparently, we need to finish the run when running on script
            wandb.finish()
       
##
    def log_traindata(self, train_loader):
        slices = self.get_slice_image(train_loader.data)
        pixels = torch.vstack((torch.hstack(slices[:2]), 
                               torch.hstack(slices[2:])))
        if re.match('laplace_*', self.hyper['filter']) and self.hyper['stage'] > 1:
            self.log_detailtensor(pixels, 'Train Data')
        else:
            self.log_imagetensor(pixels, 'Train Data')
    
    def log_groundtruth(self, test_loader):
        gtdata = test_loader.data.view(self.hyper['channels'], 
                                       -1).permute((1, 0))
        slices = self.get_slice_image(test_loader.data)
        
        pixels = torch.vstack((torch.hstack(slices[:2]), 
                               torch.hstack(slices[2:])))
        if re.match('laplace_*', self.hyper['filter']) and self.hyper['stage'] > 1:
            self.log_detailtensor(pixels, 'Ground Truth')
        else:
            self.log_imagetensor(pixels, 'Ground Truth')
        
        self.log_fft(slices, 'FFT Ground Truth')

        # if self.visualize_gt_grads:

        #     try:
        #         gt_grads = test_loader.sampler.img_grad
        #         self.log_gradmagnitude(gt_grads, 'Ground Truth - Gradient')
        #     except:
        #         print(f'No gradients in sampler and visualization is True. Set visualize_grad to False')
        
        return gtdata
    
    def log_prediction(self, model, test_loader, device):
        dims = test_loader.sampler.data_shape()
        channels = self.hyper['channels']
        domain = test_loader.sampler.coords.permute(1, 0).view(
            channels, *dims
        )
        
        domain_slices =[domain[:, self.x_slice, :, :],
                        domain[:, :, self.y_slice, :],
                        domain[:, :, self.z_slice, :],
                        domain[:, :, :, self.w_slice]]
        pred_slices = []
        grads = []
        for slice in domain_slices:
            output_dict = model(
                slice.reshape(channels, -1).permute((1, 0)).to(device))
            model_out = output_dict['model_out'].clamp(0, 1)
            grads.append(gradient(model_out, 
                                  output_dict['model_in']).detach().cpu().view(dims[0], dims[1], 3))
            pred_slices.append(
                model_out.view(dims[0], dims[1], channels).detach().cpu())

        pred_pixels = torch.vstack(
            (torch.hstack(pred_slices[:2]), torch.hstack(pred_slices[2:])))
        self.log_imagetensor(pred_pixels, 'Prediction')
        self.log_fft(pred_slices, 'FFT Prediction')

        pred_grads = torch.vstack((torch.hstack(grads[:2]), 
                                   torch.hstack(grads[2:])))

        self.log_gradmagnitude(pred_grads, 'Prediction - Gradient')
        
        return output_per_batch(model, test_loader, device)
    
    # TODO: make it work with color images and non-squared images
    def get_slice_image(self, volume):
        x = volume[:, self.x_slice, :, :].permute((1, 2, 0))
        y = volume[:, :, self.y_slice, :].permute((1, 2, 0))
        z = volume[:, :, :, self.z_slice].permute((1, 2, 0))
        # random
        try:
            w = volume[:, :, :, self.w_slice].permute((1, 2, 0))
        except IndexError as e:
            w = volume[:, :, :, -1].permute((1, 2, 0))
        return [x, y, z, w]

    def log_imagetensor(self, pixels:torch.Tensor, label:str):
        image = wandb.Image(pixels.numpy())
        wandb.log({label: image})

    def log_detailtensor(self, pixels:torch.Tensor, label:str):
        pixels = (pixels + 1.0) / (2.0)
        image = wandb.Image(pixels.numpy())    
        wandb.log({label: image})
    
    def log_gradmagnitude(self, grads:torch.Tensor, label: str):
        mag = torch.sqrt(grads[:, :, 0]**2 
                         + grads[:, :, 1]**2 
                         + grads[:, :, 2]**2).numpy()
        gmin, gmax = np.min(mag), np.max(mag)
        img = Image.fromarray(255 * (mag - gmin) / gmax).convert('L')
        wandb.log({f'Gradient Magnitude - {label}': wandb.Image(img)})
    
    def log_fft(self, slices, label:str):
        if not isinstance(slices, Sequence):
            slices = [slices]
        if self.hyper['channels'] == 3:
            slices = [0.2126*s[:, :, 0] + 0.7152*s[:, :, 1] + 0.0722*s[:, :, 2]
                      for s in slices]
        imgs = []
        for pixels in slices:
            fourier_tensor = torch.fft.fftshift(
                            torch.fft.fft2(pixels.squeeze(-1)))
            magnitude = 20 * np.log(abs(fourier_tensor.numpy()) + 1e-10)
            magnitude = magnitude / np.max(magnitude)
            graymap = cm.get_cmap('gray')
            imgs.append(magnitude)
        if len(imgs) == 1:
            img = Image.fromarray(np.uint8(graymap(imgs[0]) * 255))
        else:
            magnitude = np.vstack((np.hstack(imgs[:2]), np.hstack(imgs[2:])))
            img = Image.fromarray(np.uint8(graymap(magnitude) * 255))
        wandb.log({label: wandb.Image(img)})


    def log_extrapolation(self, model, interval, dims, device='cpu'):
        w, h, d = dims
        start, end = interval[0], interval[1]
        scale = (end - start) // 2
        neww, newh, newd = int(scale * w), int(scale * h), int(scale * d)
        ext_domain = make_grid_coords((neww, newh, newd), start, end, dim=3)
        ext_domain = ext_domain.permute(1, 0).view(
                                self.hyper['channels'], neww, newh, newd)
        domain_slices =[ext_domain[:, self.x_slice, :, :],
                        ext_domain[:, :, self.y_slice, :],
                        ext_domain[:, :, self.z_slice, :],
                        ext_domain[:, :, :, self.w_slice]]
        pred_slices = []
        for slice in domain_slices:
            values = []
            for batch in BatchSampler(
                    slice.reshape(self.hyper['channels'], -1).permute((1, 0)), 
                    self.hyper['batch_size'], drop_last=False):
                batch = torch.stack(batch)
                with torch.no_grad():
                    values.append(
                        model(batch.to(device))['model_out'].clamp(0, 1))
            values = torch.concat(values)
            pred_slices.append(values.view(neww, newh, self.hyper['channels']))
        pred_pixels = torch.vstack((torch.hstack(pred_slices[:2]), 
                                    torch.hstack(pred_slices[2:])))
        self.log_imagetensor(pred_pixels, 'Extrapolation')

    def log_point_cloud(self, model, device):
        if self.hyper.get('test_mesh', ""):
            mesh = trimesh.load_mesh(os.path.join(self.basedir, MESHES_DIR, 
                                                  self.hyper['test_mesh']))
            point_cloud, _ = trimesh.sample.sample_surface(
                                    mesh, self.hyper['ntestpoints'])
            point_cloud = torch.from_numpy(point_cloud).float()
            # center at the origin and rescale to fit a sphere of radius 0.8
            point_cloud = point_cloud - torch.mean(point_cloud, dim=0)
            scale = 0.8 / torch.max(torch.abs(point_cloud))
            point_cloud = scale * point_cloud
        else:
            point_cloud = torch.rand((self.hyper['ntestpoints'], 3))
            point_cloud = (point_cloud / torch.linalg.vector_norm(
                                    point_cloud, dim=-1).unsqueeze(-1)) * 0.8
        colors = []
        for batch in BatchSampler(point_cloud, 
                                  self.hyper['batch_size'], drop_last=False):
            batch = torch.stack(batch)
            with torch.no_grad():
                colors.append(model(batch.to(device))['model_out'])
        colors = torch.concat(colors) * 255
        if self.hyper['channels'] == 1:
            colors = torch.concat([colors, colors, colors], 1)
        
        point_cloud = torch.concat((point_cloud, colors), 1)

        wandb.log({"point_cloud": wandb.Object3D(point_cloud.numpy())})