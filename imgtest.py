import torch
import numpy as np
from PIL import Image
import torchvision.transforms as TF
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.ndimage
from src.networks.mrnet import MRFactory
from src.datasets.imagesignal import make2Dcoords

def load_image(path:str):
    img = Image.open(path).convert('L')
    transf = TF.Compose([TF.ToTensor()])
    return torch.flatten(transf(img))

def as_imagetensor(tensor):
    w = int(np.sqrt(len(tensor)))
    h = w
    print(w, 'x', h)
    pixels = tensor.cpu().detach().unflatten(0, (w, h))
    return pixels

def log_imagetensor(pixels:torch.Tensor, label:str):
    img = Image.fromarray(np.uint8(pixels.numpy() * 255))
    img.save(f'img/{label}.png')

def log_fft(pixels:torch.Tensor, label:str):
    print(pixels.shape)
    fourier_tensor = torch.fft.fftshift(torch.fft.fft2(pixels))
    magnitude = 20 * np.log(abs(fourier_tensor.numpy()))
    print(magnitude)
    # plt.imsave(f'img/{label}2.png', magnitude, cmap = 'gray')
    magnitude = magnitude / np.max(magnitude)
    graymap = cm.get_cmap('gray')
    img = Image.fromarray(np.uint8(graymap(magnitude) * 255))
    img.save(f'img/{label}.png')
    
def luminance_tensor(path):
    img = Image.open(path)
    transf = TF.Compose([TF.ToTensor()])
    return torch.flatten(transf(img))

def gradient_tensor(img):
    grads_x = scipy.ndimage.sobel(img.numpy(), axis=0)[..., None]
    grads_y = scipy.ndimage.sobel(img.numpy(), axis=1)[..., None]
    grads_x, grads_y = torch.from_numpy(grads_x), torch.from_numpy(grads_y)
    return torch.stack((grads_x, grads_y), dim=-1).view(-1, 2)

def gradient_mag(img):
    grads_x = scipy.ndimage.sobel(img.numpy(), axis=0)
    grads_y = scipy.ndimage.sobel(img.numpy(), axis=1)
    return np.hypot(grads_x, grads_y)

def test_loaded_model(modelpath):
    model = MRFactory.load_state_dict(modelpath)
    coords = make2Dcoords(513, 513)
    output = model(coords)['model_out']
    i = 0
    for v in output:
        if v < 0.0 or v > 1.0:
            print(v)
            i = i + 1
    print('HOP!', i)
    pixels = as_imagetensor(torch.clamp(output, 0, 1)).squeeze(-1)
    # with open('E:\Workspace\impa\siren-song\models\leia.txt', 'w') as f:
    #     print(pixels, f)
    # print('SHAE', pixels.shape)
    img = Image.fromarray(np.uint8(pixels.numpy() * 255))
    img.show()
    # log_imagetensor(pixels, 'TESTLOAD')
    # with open('E:\Workspace\impa\siren-song\models\loaded.json', 'w') as f:
    #     for name, param in model.named_parameters():
    #         if param.requires_grad:
    #             print(name, param.data, file=f)
    

if __name__ == '__main__': 
    test_loaded_model('models\MPchec_3-3_w24T_hf96_MEp7_hl1_d0_1025px.pth')
    # img = Image.open('img/lena.png')
    # tensor = load_image('img/lena.png')
    # print(tensor.shape, tensor[0].view(-1))
    # pixels = as_imagetensor(tensor)
    # print(pixels.shape)
    # gradtensor = gradient_tensor(pixels).unflatten(0, (512, 512))
    # mag = np.hypot(gradtensor[:, :, 0].squeeze(-1).numpy(),
    #                 gradtensor[:, :, 1].squeeze(-1).numpy())
    # # mag = gradient_mag(pixels)
    # print('MAG shape:', mag.shape)
    # gmin, gmax = np.min(mag), np.max(mag)
    # print('MinMax:', gmin, gmax)
    # Image.fromarray(255 * mag / gmax).convert('L').save('img/mag3.png')