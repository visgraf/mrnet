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
    tensor_filt_decimated = to_tensor(pil_filtered_decimated)
    return ImageSignal(tensor_filt_decimated,
                        w_new, h_new,
                        None,
                        signal.channels,
                        useattributes=signal._useattributes)

def pyrup2d_opencv(image,num_times,dims_to_upscale):
    img_scale = image
    for level in range(num_times):
        img_scale = cv2.pyrUp(img_scale,dstsize=dims_to_upscale[level])

    img_scale = cv2.pyrUp(img_scale,dstsize=dims_to_upscale[-1])
    
    return img_scale

def pyrup2D_imagesignal(signal,num_times,dims_to_upscale):

    img_pil = signal.image_pil()
    img_npy = pil2opencv(img_pil)    

    scaled_up_image = pyrup2d_opencv(img_npy,num_times,dims_to_upscale)
    pil_scaled_up_image = opencv2pil(scaled_up_image)

    w_new, h_new = pil_scaled_up_image.size
    tensor_scaled_up_image = to_tensor(pil_scaled_up_image)
    return ImageSignal(tensor_scaled_up_image,
                        w_new, h_new,
                        None,
                        signal.channels,
                        useattributes=signal._useattributes)

def construct_gaussian_pyramid2D(signal, num_levels):
    pyramid = [signal]
    print(signal.dimensions())
    for _ in range(num_levels-1):
        signal = pyrdown2D(signal)
        print(signal.dimensions())
        pyramid.append(signal)
    return pyramid


def construct_gaussian_tower(gaussian_pyramid):
    pyramid_dimensions = [signal_dims.dimensions() for signal_dims in gaussian_pyramid[:-1] ]
    gauss_tower=[gaussian_pyramid[0]]
    for level,signal in enumerate(gaussian_pyramid[1:]):
        dims_to_upscale = pyramid_dimensions[:(level+1)]
        dims_to_upscale.reverse()
        signal = pyrup2D_imagesignal(signal,level,dims_to_upscale)
        print(signal.dimensions())
        gauss_tower.append(signal)
    return gauss_tower

def create_MR_structure(img_signal,num_levels,type_pyr="pyramid"):

    gaussian_pyramid = construct_gaussian_pyramid2D(img_signal,num_levels)

    if type_pyr=="pyramid":
        return gaussian_pyramid

    gaussian_tower = construct_gaussian_tower(gaussian_pyramid)

    if type_pyr=="tower":
        return gaussian_tower



    



    