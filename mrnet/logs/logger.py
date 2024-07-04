import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import wandb
import yaml
import trimesh
from datetime import datetime as dt
from PIL import Image
from typing import Sequence
from mrnet.datasets.utils import (output_on_batched_dataset, 
                            output_on_batched_grid, rgb_to_grayscale, ycbcr_to_rgb, RESIZING_FILTERS, INVERSE_COLOR_MAPPING)
from mrnet.networks.mrnet import MRNet, MRFactory


def make_runname(hyper, name):
    w0 = f"w{hyper['omega_0']}{'T' if hyper['superposition_w0'] else 'F'}"
    hl = f"hl{hyper['hidden_layers']}"
    epochs = f"Ep{hyper['max_epochs_per_stage']}"
    if isinstance(hyper['hidden_features'], Sequence):
        hf = ''.join([str(v) for v in hyper['hidden_features']])
    else:
        hf = hyper['hidden_features']
    hf = f"hf{hf}"
    period = hyper.get('period', 0)
    per = f"pr{period}" if period > 0 else ""
    stage = f"{hyper['stage']}-{hyper['max_stages']}"
    if name:
        return f"{name}_{stage}_{w0}{per}_{hl}_{hf}_{epochs}"
    return f"{stage}_{w0}{per}_{hl}_{hf}_{epochs}"

def get_incremental_name(path):
    names = [nm.split()[0][-3:] for nm in os.listdir(path)]
    if not names:
        return 1
    names.sort()
    try:
        return int(names[-1]) + 1
    except ValueError:
        return 1
    
def slugfy(text):
    return '_'.join(text.lower().split())

class Logger:
    def __init__(self, project: str,
                        name: str, 
                        hyper: dict,
                        basedir: str, 
                        **kwargs):
        
        self.project = project
        self.name = name
        self.hyper = hyper
        self.basedir = basedir
        self.stage = kwargs.get('stage', 0)


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
    
    def log_graph(self, Xs, Ys, label, **kwargs):
        raise NotImplementedError()


class LocalLogger(Logger):

    def make_dirs(self):
        for key in self.subpaths:
            os.makedirs(os.path.join(self.savedir, self.subpaths[key]), exist_ok=True)
    
    def get_path(self, category):
        path = os.path.join(self.savedir, self.subpaths[category])
        os.makedirs(path, exist_ok=True)
        return path

    def __init__(self, project: str, 
                        name: str, 
                        hyper: dict,
                        basedir: str, 
                        **kwargs):
        super().__init__(project, name, hyper, 
                         basedir, **kwargs)
        
        try: 
            logs_path = hyper['logs_path']
        except KeyError:
            logs_path = basedir.joinpath('logs')
        self.subpaths = {
            "loss": "",
            "attr": "attr",
            "models": "models",
            "gt": "gt",
            "pred": "pred",
            "etc": "etc",
            "zoom": "zoom"
        }

        try:
            self.runname = hyper['run_name']
        except KeyError:
            self.runname = make_runname(hyper, "")

        now = dt.now()
        timetag = now.strftime("%Y%m%d")
        name = f"{timetag}_{name}"
        self.savedir = os.path.join(logs_path, name, 
                                    f"{now.strftime('%H%M')}st{self.runname}")

    @property
    def loss_filepath(self):
        return os.path.join(self.savedir, 'losses.csv')

    def prepare(self, model):
        self.make_dirs()
        hypercontent = yaml.dump(self.hyper)
        with open(os.path.join(self.savedir, "hyper.yml"), "w") as hyperfile:
            hyperfile.write(hypercontent)

    def log_losses(self, log_dict:dict):
        file_exists = os.path.isfile(self.loss_filepath)
        with open(self.loss_filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=log_dict.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(log_dict)
            
    def log_images(self, pixels, label, captions=None, **kwargs):
        pixels, captions = super().log_images(pixels, label, captions)
        path = self.get_path(kwargs['category'])

        for i, image in enumerate(pixels):
            try:
                filename = kwargs["fnames"][i]
            except (KeyError, IndexError):
                slug = slugfy(label)
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

    def log_graph(self, Xs, Ys, label, **kwargs):
        path = self.get_path(kwargs['category'])
        try:
            filename = kwargs["fnames"]
        except KeyError:
            filename = slugfy(label)

        if not isinstance(Xs, Sequence):
            Xs = [Xs] * len(Ys)
        
        captions = kwargs.get('captions', '')
        marker = kwargs.get('marker', ['', '', '', ''])
        style = kwargs.get('linestyle', ['-', '--', '-.', ':'])
        width = kwargs.get('linewidth', [2] * 4)
        fig, ax = plt.subplots()
        for i, (x, y) in enumerate(zip(Xs, Ys)):
            ax.plot(x, y, 
                    label=captions[i], 
                    linestyle=style[i % len(style)], 
                    linewidth=width[i % len(width)], 
                    marker=marker[i % len(marker)])
        ax.set_title(label)
        ax.set_xlabel(kwargs.get('xname', 'coords'))
        # ax.set_aspect('equal')
        ax.grid(True, which='both')
        # seaborn.despine(ax=ax, offset=0)
        ax.legend()
        fig.savefig(os.path.join(path, filename))
        plt.close()


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

    def finish(self):
        columns = {}
        with open(self.loss_filepath) as f:
            reader = csv.DictReader(f)
            captions = list(reader.fieldnames)
            for row in reader:
                for key in captions:
                    value = float(row[key])
                    try:
                        columns[key].append(value)
                    except KeyError:
                        columns[key] = [value]
        x = list(range(len(columns[captions[0]])))
        for loss in captions:
            self.log_graph([x], [columns[loss]], 
                           loss,
                           category='loss',
                           captions=[loss.split()[0]],
                           xname='epochs')
        ys = list(columns.values())
        xs = [x] * len(ys)
        self.log_graph(xs, ys, 
                       "All losses", 
                       captions=captions,
                       category='loss',
                       fname='all_losses',
                       xname='epoch')
        print(f"All results logged to {self.savedir}")
            

class WandBLogger(Logger):

    def __init__(self, project: str, name: str, hyper: dict, basedir: str, **kwargs):
        super().__init__(project, name, hyper, basedir, **kwargs)
        self.entity = kwargs.get('entity', None)
        self.config = kwargs.get('config', None) 
        self.settings = kwargs.get('settings', None)

        try:
            self.runname = hyper['run_name']
        except KeyError:
            self.runname = make_runname(hyper, name)

    def prepare(self, model):
        wandb.finish()
        wandb.init(project=self.project, 
                    entity=self.entity, 
                    name=self.runname, 
                    config=self.hyper,
                    settings=self.settings)
        wandb.watch(model, log_freq=10, log='all')

    def log_losses(self, log_dict):
        wandb.log(log_dict)

    def log_images(self, pixels, label, captions=None, **kwargs):
        pixels, captions = super().log_images(pixels, label, captions)
        if isinstance(pixels[0], torch.Tensor):
            pixels = [p.squeeze(-1).numpy() for p in pixels]
        images = [wandb.Image(pixels[i],
                              caption=captions[i]) for i in range(len(pixels))]
        wandb.log({label: images})

    def log_graph(self, Xs, Ys, label, **kwargs):
        captions = kwargs['captions']
        name = kwargs.get('fname', slugfy(label))
        # embed()
        wandb.log({
            name: wandb.plot.line_series(
                xs = Xs,
                ys = Ys,
                keys=captions,
                title=label,
                xname=kwargs.get('xname', 'coords'),
            )
        })

    def log_metric(self, metric_name, value, label):
        table = wandb.Table(data=[(label, value)], 
                            columns = ["Stage", metric_name.upper()])
        wandb.log({
            f"{metric_name}_value" : wandb.plot.bar(
                                table, "Stage", metric_name.upper(),
                                title=f"{metric_name} for reconstruction")})
        
    def log_model(self, model, **kwargs):
        temp_path = os.path.join(self.basedir, kwargs['path'])
        os.makedirs(temp_path, exist_ok=True)
        filename = kwargs.get('fname', 'final')
        logpath = os.path.join(temp_path, filename + '.pth')
        MRFactory.save(model, logpath)

        artifact = wandb.Artifact(filename, type='model')
        artifact.add_file(logpath)
        wandb.log_artifact(artifact)

    def finish(self):
        wandb.finish()