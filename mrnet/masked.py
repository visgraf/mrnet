import random
from PIL import Image
import os
from pathlib import Path
import torch
from logs.wandblogger import WandBLogger2D
from training.trainer import MRTrainer
from datasets.signals import ImageSignal#, make_mask
from networks.mrnet import MRFactory
from datasets.pyramids import create_MR_structure
import yaml
from yaml.loader import SafeLoader


def sanity_mask(res):
    Image.new('L', (res, res), 255).save(f'img/masks/sanity{res}.png')

def find_correspondents(pixel, size, period):
    x, y = pixel
    p, q = (x + period) % size,(y + period) % size
    return set([(p, y),
                (x, q),
                (p, q),
                (x, y)])

def random_mask_non_contiguous(res):
    width = height = res
    valid_pixels = set([(i, j) for i in range(height) for j in range(width)])
    selected = []
    period = width//2
    print(len(valid_pixels))
    while len(valid_pixels) > 0:
        px = random.choice(list(valid_pixels))
        correspondents = find_correspondents(px, width, period)
        valid_pixels = valid_pixels - correspondents
        selected.append(px)

    mask = Image.new('L', (width, height))
    for px in selected:
        mask.putpixel((px), 255)
    mask.save(f"img/mask{res}.png")
    print(len(selected), "selected pixels")

def shape_mask(res):
    mask = Image.new('L', (res, res))
    # diagonal
    for i in range(res):
        for j in range(res - 1, i, -1):
            mask.putpixel((i, res-j-1), 255)
    mask.save("img/masks/shapemask.png")
    
def main():

    os.environ["WANDB_NOTEBOOK_NAME"] = "train-wb.ipynb"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    BASE_DIR = Path('.').absolute().parents[0]
    IMAGE_PATH = BASE_DIR.joinpath('img')
    MODEL_PATH = BASE_DIR.joinpath('models')
    torch.manual_seed(777)

    #-- hyperparameters in configs --#
    config_file = '../configs/siggraph_asia/config_siggraph_masked.yml'
    with open(config_file) as f:
        hyper = yaml.load(f, Loader=SafeLoader)
        if isinstance(hyper['batch_size'], str):
            hyper['batch_size'] = eval(hyper['batch_size'])
        if hyper.get('channels', 0) == 0:
                hyper['channels'] = hyper['out_features']
        print(hyper)
    imgpath = os.path.join(IMAGE_PATH, hyper['image_name'])
    project_name = hyper.get('project_name', 'dev_sandbox')
    maskpath = None
    # maskpath = "/Users/hallpaz/Workspace/impa/mrimg/img/synthetic/mask_inverted.png" #make_mask(imgpath, hyper['mask_color'])
    hyper['device']


    base_signal = ImageSignal.init_fromfile(
                        imgpath,
                        domain=hyper['domain'],
                        channels=hyper['channels'],
                        sampling_scheme=hyper['sampling_scheme'],
                        width=hyper['width'], height=hyper['height'],
                        attributes=hyper['attributes'],
                        batch_size=hyper['batch_size'],
                        color_space=hyper['color_space'])

    train_dataset = create_MR_structure(base_signal, 
                                        hyper['max_stages'], 
                                        hyper['filter'], 
                                        hyper['decimation'],
                                        hyper['pmode'])
    test_dataset = create_MR_structure(base_signal, 
                                        hyper['max_stages'], 
                                        hyper['filter'], 
                                        False,
                                        hyper['pmode'])

    if hyper['width'] == 0:
        hyper['width'] = base_signal.shape[-1]
    if hyper['height'] == 0:
        hyper['height'] = base_signal.shape[-1]

    img_name = os.path.basename(hyper['image_name'])
    mrmodel = MRFactory.from_dict(hyper)
    print("Model: ", type(mrmodel))
    wandblogger = WandBLogger2D(project_name,
                                f"{hyper['model']}{hyper['filter'][0].upper()}{img_name[0:5]}",
                                hyper,
                                BASE_DIR)
    mrtrainer = MRTrainer.init_from_dict(mrmodel, 
                                        train_dataset, 
                                        test_dataset, 
                                        wandblogger, 
                                        hyper)
    mrtrainer.train(hyper['device'])


def evaluate_model(path, interval=None):
    model = MRFactory.load_state_dict(path).to('cuda')
    # 6 x 3 -2, 2 = 550 + 275 825
    a = [0, 0]
    b = [3, 3]
    size = (abs(b[0] - a[0]) * 275, abs(b[1] - a[1]) * 275)
    print(size)
    grid = make_grid_coords((size[0], size[1]), a, b, 2)
    print(grid[..., 0].min(), grid[..., 0].max(), grid[..., 1].min(), grid[..., 1].max())
    with torch.no_grad():
        pixels = output_on_batched_points(model, grid, 128 * 256, 'cuda')
    img = (ycbcr_to_rgb(pixels).clamp(0, 1).cpu() * 255
           ).reshape((size[1], size[0], 3)).numpy().astype(np.uint8)
    img = Image.fromarray(img)
    img.save("image.png")

if __name__ == '__main__':
    from datasets.utils import make_grid_coords, output_on_batched_points, ycbcr_to_rgb
    import numpy as np
    from PIL import Image
    evaluate_model("models/siggraph/shape.pth")
    # res = 275
    # sanity_mask(res)
    # for i in [2, 4]:
    # random_mask_non_contiguous(138)
    # shape_mask(res)
