from mrnet.networks.mrnet import MRNet, MRFactory
from .logger import Logger

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
from mrnet.training.loss import gradient
from typing import Sequence, Tuple
import yaml
import skimage

from IPython import embed

class LocalLogger(Logger):
    def __init__(self, project: str, 
                        name: str, 
                        hyper: dict,
                        basedir: str, 
                        entity=None, 
                        config=None, 
                        settings=None,
                        to_file=False):
        super().__init__(project, name, hyper, 
                         basedir, entity, config, settings)
        self.logs = {}
        logs_path = basedir.joinpath('logs')
        self.modelspath = os.path.join(logs_path, project, 'models')
        self.to_file = to_file
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
                            
        self.log_PSNR(gt, pred, stage_number)
        #if stage_number == self.hyper['max_stages']:
        self.log_SSIM(gt, pred, stage_number)
        
    def on_train_finish(self, trained_model, total_epochs):
        if self.to_file:
            filename = os.path.join(self.savedir, "evalmetrics.csv")
            psnr = self.logs[self.hyper['max_stages']]['psnr']
            ssim = self.logs[self.hyper['max_stages']]['ssim']
            mse = self.logs[self.hyper['max_stages']]['mse']
            with open(filename, "a") as psnrfile:
                psnrfile.write(f"{os.path.basename(self.hyper['image_name'])}, {psnr}, {ssim}, {mse}\n")

            filename = os.path.join(self.savedir, "parameters.csv")
            param_size = 0
            for param in trained_model.parameters():
                param_size += param.nelement()
            buffer_size = 0
            for buffer in trained_model.buffers():
                buffer_size += buffer.nelement()
            first_write = False
            if not os.path.isfile(filename):
                first_write = True
            with open(filename, "a") as paramfile:
                if first_write:
                    paramfile.write("n_parameters, n_buffers, n_epochs\n")
                paramfile.write(f"{param_size}, {buffer_size}, {total_epochs}\n")
        print(f'Training finished after {total_epochs} epochs')
        
        filename = f"{self.hyper['model']}{self.hyper['image_name'][0:4]}.pth"
        path = os.path.join(self.models_path, filename)
        MRFactory.save(trained_model, path)

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
        model_out = torch.clamp(output_dict['model_out'], 0.0, 1.0).cpu()

        img = self.tensor_to_img(model_out)
        if self.to_file:
            filename = os.path.join(self.savedir, "pred", f"pred{stage_number}_{os.path.basename(self.hyper['image_name'])}")
            img.save(filename)
        else:
            self.logs[stage_number]['pred'] = img
        return model_out

    def log_PSNR(self, gt, pred, stage_number):
        mse = torch.mean(gt - pred)**2
        psnr = 10*torch.log10(1 / (mse + 1e-10))
        print(self.logs)
        self.logs[stage_number]['psnr'] = psnr.item()
        self.logs[stage_number]['mse'] = mse.item()

    def log_SSIM(self, gt, pred, stage_number):
        ssim = skimage.metrics.structural_similarity(gt.detach().cpu().numpy(), 
                                                    pred.detach().cpu().numpy(),
                                                    data_range=1, channel_axis=-1)
        self.logs[stage_number]['ssim'] = ssim

