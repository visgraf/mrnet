from decimal import InvalidOperation
import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from .imagesignal import ImageSignal
from .constants import Sampling
import cv2
import numpy as np
from PIL import Image


def pil2opencv(pil_image):
    
    open_cv_image = np.array(pil_image) 

    return open_cv_image

def opencv2pil(numpy_image):
    
    im_pil = Image.fromarray(numpy_image)
    
    return im_pil

def pyrdown2D(signal, decimate=True):

    img_pil = signal.image_pil()
    img_npy = pil2opencv(img_pil)
    transf = ToTensor()

    

    w, h = signal.dimensions()
    if decimate:
        filtered_decimated = cv2.pyrDown(img_npy)
        pil_filtered_decimated = opencv2pil(filtered_decimated)
        w_new, h_new = pil_filtered_decimated.size

        tensor_filt_decimated = transf(pil_filtered_decimated)
        return ImageSignal(tensor_filt_decimated,
                            w_new, h_new,
                            None,
                            signal.channels,
                            useattributes=signal._useattributes)
    
    else:
        gauss_blur = cv2.GaussianBlur(img_npy,(5,5),0)
        pil_gauss_blur = opencv2pil(gauss_blur)
        tensor_gauss_blur = transf(pil_gauss_blur)
        return ImageSignal(tensor_gauss_blur,
                            w, h,
                            signal.coordinates,
                            signal.channels,
                            useattributes=signal._useattributes)

def gaussian_pyramid2D(signal, levels):
    pyramid = [signal]
    for s in range(levels-1):
        signal = pyrdown2D(signal)
        pyramid.append(signal)
    return pyramid

def gaussian_tower2D(signal, levels):
    '''Only works with box filter for now'''
    
    tower = [signal]
    for s in range(levels-1):
        signal = pyrdown2D(signal, decimate=False)
        tower.append(signal)
    return tower