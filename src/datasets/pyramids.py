import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor
from .imagesignal import ImageSignal
import cv2
import numpy as np
from PIL import Image

def pil2opencv(pil_image):
        
        open_cv_image = np.array(pil_image) 
        return open_cv_image

def opencv2pil(numpy_image):
        
        im_pil = Image.fromarray(numpy_image)
        return im_pil

def pyrdown2D(signal):

    img_pil = signal.image_pil()
    img_npy = pil2opencv(img_pil)    

    filtered_decimated = cv2.pyrDown(img_npy)
    pil_filtered_decimated = opencv2pil(filtered_decimated)

    w_new, h_new = pil_filtered_decimated.size
    print(f'pyrdown result: w:{w_new},h:{h_new}')
    tensor_filt_decimated = to_tensor(pil_filtered_decimated)
    return ImageSignal(tensor_filt_decimated,
                        w_new, h_new,
                        None,
                        signal.channels,
                        useattributes=signal._useattributes)

def pyrup2d_opencv(image,num_times,orig_w,orig_h):
    img_scale = image
    for _ in range(num_times):
        img_scale = cv2.pyrUp(img_scale)

    img_scale = cv2.pyrUp(img_scale)
    
    return img_scale

def pyrup2D_imagesignal(signal,num_times,orig_w,orig_h):

    img_pil = signal.image_pil()
    img_npy = pil2opencv(img_pil)    

    scaled_up_image = pyrup2d_opencv(img_npy,num_times,orig_w,orig_h)
    pil_scaled_up_image = opencv2pil(scaled_up_image)

    w_new, h_new = pil_scaled_up_image.size
    print(f'pyrup result: w:{w_new},h:{h_new}')
    tensor_scaled_up_image = to_tensor(pil_scaled_up_image)
    return ImageSignal(tensor_scaled_up_image,
                        w_new, h_new,
                        None,
                        signal.channels,
                        useattributes=signal._useattributes)

def construct_gaussian_pyramid2D(signal, num_levels):
    pyramid = [signal]
    for _ in range(num_levels-1):
        signal = pyrdown2D(signal)
        pyramid.append(signal)
    return pyramid

def construct_gaussian_tower(gaussian_pyramid,orig_w,orig_h):
    gauss_tower=[gaussian_pyramid[0]]
    for level,signal in enumerate(gaussian_pyramid[1:]):
        signal = pyrup2D_imagesignal(signal,level,orig_w,orig_h)
        gauss_tower.append(signal)
    return gauss_tower

def create_MR_structure(img_signal,num_levels,type_pyr="pyramid"):

    gaussian_pyramid = construct_gaussian_pyramid2D(img_signal,num_levels)

    if type_pyr=="pyramid":
        return gaussian_pyramid

    orig_w, orig_h = img_signal.dimensions()
    gaussian_tower = construct_gaussian_tower(gaussian_pyramid,orig_w,orig_h)

    if type_pyr=="tower":
        return gaussian_tower



    



    