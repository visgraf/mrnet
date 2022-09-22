

class Logger:
    def __init__(self) -> None:
        pass

    def on_train_start(self):
        pass

    def on_train_finish(self, trained_model, total_epochs):
        pass

    def on_epoch_finish(self, current_model, epochloss):
        pass

    def on_batch_finish(self, batchloss):
        pass

    def on_stage_start(self, current_model, stage_number):
        pass
    
    def on_stage_trained(self, current_model, train_loader, test_loader):
        pass

    