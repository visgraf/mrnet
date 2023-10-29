import yaml

from torchvision.transforms.functional import to_tensor, to_pil_image
from mrnet.datasets.sampler import RegularSampler
from mrnet.datasets.signals import ImageSignal
from PIL import Image


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
