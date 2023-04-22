import wandb
import torch
import os
import re
import numpy as np
import torch.utils.data as data_utils

from scipy.fft import fft, fftfreq
from matplotlib import cm
from PIL import Image
from copy import deepcopy
from pathlib import Path

import warnings
from training.loss import gradient
from datasets.sampler import make_grid_coords

from .logger import Logger
from networks.mrnet import MRNet, MRFactory

MODELS_DIR = 'models'

class WandBLogger(Logger):
    def __init__(self, project: str, 
                        name: str, 
                        hyper: dict,
                        basedir: str, 
                        entity=None, 
                        config=None, 
                        settings=None,
                        visualize_gt_grads=False):
        super().__init__(project, name, hyper, 
                         basedir, entity, config, settings)
        self.visualize_gt_grads = visualize_gt_grads

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



class WandBLogger1D(WandBLogger):

    def on_stage_start(self, current_model, stage_number, updated_hyper=None):
        stage_hyper = deepcopy(self.hyper)
        if updated_hyper:
            for key in updated_hyper:
                stage_hyper[key] = updated_hyper[key]
        runname = f"{self.name}{stage_hyper['stage']}/{stage_hyper['max_stages']}_w{stage_hyper['omega_0']}{'T' if stage_hyper['superposition_w0'] else 'F'}_hf{stage_hyper['hidden_features']}"
        wandb.init(project=self.project, 
                    entity=self.entity, 
                    name=runname, 
                    config=stage_hyper,
                    settings=self.settings)
        wandb.watch(current_model, log_freq=10, log='all')

    def on_stage_trained(self, current_model: MRNet,
                                train_loader, test_loader):
        super().on_stage_trained(current_model, train_loader, test_loader)

        # testX, testY = next(iter(test_loader))
        # # we'll need samples sorted for FFT
        # # xtensor, sortidx = torch.sort(testX)
        # if isinstance(testY, dict):
        #     testY = testY['d0']

        
        # with torch.no_grad():
        #     fulloutput = current_model(testX.to(device))

        # pred = fulloutput['model_out']
        
        # self.log_fft(testY, pred)

        # self.log_results(current_model, train_loader, test_loader)
        
        # current_model.train()
        

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

    def log_results(self, model, train_loader, test_loader):
        trainX, trainY = next(iter(train_loader))
        testX, testY = next(iter(test_loader))

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
        device = self.hyper.get('device', 'cpu')
        model.eval()
        output_dict = model(trainX.to(device))
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
                                    X_test, Y_test)

        
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

    def log_extrapolation(self, model, interval, X_test, Y_test):
        space = X_test[1] - X_test[0]
        start, end = interval[0], interval[1]
        newsamplesize = abs(int((end - start) / space)) + 1
        ext_x = torch.linspace(start, end, newsamplesize)
        out_dict = model(ext_x.view(-1, 1).to(self.hyper.get('device', 'cpu')))
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
                                train_loader,
                                test_loader):
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
                                    test_loader.dimensions(), device)

        current_model.train()
        current_model.to(self.hyper['device'])
        
        if current_model.n_stages() < self.hyper['max_stages']:
            # apparently, we need to finish the run when running on script
            wandb.finish()
       
##
    def log_traindata(self, train_loader):
        traindata = train_loader.data.view(-1, self.hyper['channels'])
        pixels = self.as_imagetensor(torch.clamp(traindata, 0, 1))

        if re.match('laplace_*', self.hyper['filter']) and self.hyper['stage'] > 1:
            self.log_detailtensor(pixels, 'Train Data')
        else:
            self.log_imagetensor(pixels, 'Train Data')
    
    def log_groundtruth(self, test_loader):
        gtdata = test_loader.data.view(-1, self.hyper['channels'])
        pixels = self.as_imagetensor(torch.clamp(gtdata, 0, 1))
        
        if re.match('laplace_*', self.hyper['filter']) and self.hyper['stage'] > 1:
            self.log_detailtensor(pixels, 'Ground Truth')
        else:
            self.log_imagetensor(pixels, 'Ground Truth')
        
        self.log_fft(pixels, 'FFT Ground Truth')

        if self.visualize_gt_grads:

            try:
                gt_grads = test_loader.sampler.img_grad
                self.log_gradmagnitude(gt_grads, 'Ground Truth - Gradient')
            except:
                print(f'No gradients in sampler and visualization is True. Set visualize_grad to False')
        
        return gtdata

    def log_prediction(self, model, test_loader, device):
        output_dict = model(test_loader.sampler.coords.to(device))
        model_out = torch.clamp(output_dict['model_out'], 0.0, 1.0)

        pred_pixels = self.as_imagetensor(model_out)
        self.log_imagetensor(pred_pixels, 'Prediction')
        self.log_fft(pred_pixels, 'FFT Prediction')

        model_grads = gradient(model_out, output_dict['model_in'])
        pred_grads = torch.reshape(model_grads, (-1, 2))
        self.log_gradmagnitude(pred_grads, 'Prediction - Gradient')
        
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
        start, end = interval[0], interval[1]
        scale = (end - start) // 2
        neww, newh = int(scale * w), int(scale * h)
        
        ext_domain = make_grid_coords((neww, newh), start, end, dim=2)

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