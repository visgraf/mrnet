import numpy as np
import time
from mrnet.logs.handler import ResultHandler
from mrnet.logs.logger import LocalLogger, Logger, WandBLogger
from mrnet.networks.mrnet import MRNet


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
        self.handler.finish()

    def on_batch_finish(self, batchloss):
        pass

    def on_epoch_finish(self, current_model, epochloss):
        self.handler.log_losses(epochloss)

    def on_train_finish(self, trained_model, total_epochs):
        print(f'Training finished after {total_epochs} epochs')

