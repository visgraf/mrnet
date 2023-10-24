import yaml
from yaml.loader import SafeLoader
from mrnet.training.optimizer import (OptimizationHandler,
                                      MirrorOptimizationHandler)

from torchvision.transforms.functional import to_tensor, to_pil_image
from mrnet.datasets.sampler import RegularSampler
from mrnet.datasets.signals import ImageSignal
from PIL import Image


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
    
def make_mask(srcpath, mask_color):
    img = np.array(Image.open(srcpath))
    mask = img != mask_color
    path = Path(srcpath)
    path = path.parent.absolute().joinpath("mask.png")
    Image.fromarray(mask).save(path)
    return str(path)


def init_fromfile(imagepath, 
                      domain=[-1, 1],
                      attributes={},
                      batch_size=0,
                      color_space='RGB',
                      sampler_class=RegularSampler,
                      **kwargs):
        img = Image.open(imagepath)
        img.mode
        if color_space != img.mode:
            img = img.convert(color_space)

        width = kwargs.get('width', 0)
        height = kwargs.get('height', 0)
        if width or height:
            if not height:
                height = img.height
            if not width:
                width = img.width
            img = img.resize((width, height))
        img_tensor = to_tensor(img)

        return ImageSignal(img_tensor,
                            domain=domain,
                            attributes=attributes,
                            SamplerClass=sampler_class,
                            batch_size=batch_size,
                            color_space=color_space)