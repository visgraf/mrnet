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
    per = f"pr{hyper['period']}"
    timetag = dt.now().strftime("%Y%m%d-%H%M")
    return f"{name}_{stage}_{w0}_{hf}_{epochs}_{hl}_{per}_{timetag}"

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

    def log_losses(self, log_dict:dict):
        filepath = os.path.join(self.savedir, 'losses.csv')
        file_exists = os.path.isfile(filepath)
        with open(filepath, 'a') as f:
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
        
        captions = kwargs['captions']
        marker = kwargs.get('marker', ['', ''])
        style = kwargs.get('linestyle', ['-', '--'])
        width = kwargs.get('linewidth', [2, 2])
        fig, ax = plt.subplots()
        for i, (x, y) in enumerate(zip(Xs, Ys)):
            ax.plot(x, y, 
                    label=captions[i], 
                    linestyle=style[i], 
                    linewidth=width[i], marker=marker[i])
        ax.set_title(label)
        ax.set_xlabel(kwargs.get('xname', 'coords'))
        # ax.set_aspect('equal')
        ax.grid(True, which='both')
        # seaborn.despine(ax=ax, offset=0)
        ax.legend()
        fig.savefig(os.path.join(path, filename))


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

class WandBLogger(Logger):

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

    def log_pointcloud():
        pass

    def finish(self):
        wandb.finish()