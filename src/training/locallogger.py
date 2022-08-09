from networks.mrnet import MRNet
from .logger import Logger

import torch
from torch.utils.data import DataLoader
from training.loss import gradient
from datasets.imagesignal import ImageSignal
from typing import Sequence, Tuple


class LocalLogger2D(Logger):
    def __init__(self, hyper: dict) -> None:
        self.hyper= hyper

    def on_train_start(self):
        self.trainloss = []

    def on_epoch_finish(self, current_model, epochloss):
        self.trainloss.append(sum(epochloss.values()))

    def on_stage_start(self, current_model, stage_number, updated_hyper=None):
        pass

    def on_stage_trained(self, current_model: MRNet,
                                train_loader: DataLoader,
                                test_loader: DataLoader):
        X, Y = next(iter(test_loader))
        current_model.eval()
        ###
        print(f'Stage finished')
        current_model.train()
        

    def on_train_finish(self, trained_model, total_epochs):
        print(f'Training finished after {total_epochs} epochs')