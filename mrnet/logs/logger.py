

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

    def on_train_start(self):
        pass

    def on_train_finish(self, trained_model, total_epochs):
        pass

    def on_epoch_finish(self, current_model, epochloss):
        pass

    def on_batch_finish(self, batchloss):
        pass

    def on_stage_start(self, current_model, stage_number, updated_hyper=None):
        pass
    
    def on_stage_trained(self, current_model, train_loader, test_loader):
        current_model.to(self.hyper.get('eval_device', 'cpu'))
        current_model.eval()