import yaml
from yaml.loader import SafeLoader
from mrnet.training.optimizer import (OptimizationHandler,
                                      MirrorOptimizationHandler)

def load_hyperparameters(config_path):
    with open(config_path) as f:
        hyper = yaml.load(f, Loader=SafeLoader)
        if isinstance(hyper['batch_size'], str):
            hyper['batch_size'] = eval(hyper['batch_size'])
        if hyper.get('channels', 0) == 0:
                hyper['channels'] = hyper['out_features']
    
    return hyper

def get_optim_handler(handler_type):
    if handler_type == 'regular':
         return OptimizationHandler
    elif handler_type == 'mirror':
         return MirrorOptimizationHandler
    else:
         raise ValueError(f"Invalid handler_type")
    