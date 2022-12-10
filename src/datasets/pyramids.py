import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor
from .imagesignal import ImageSignal
import cv2
import numpy as np
from PIL import Image

def resize_half_image(numpy_image):
    dims = numpy_image.shape
    resized_img = cv2.resize(numpy_image, (dims[0]//2, dims[1]//2), interpolation = cv2.INTER_AREA)
    return resized_img

def pil2opencv(pil_image): 
        open_cv_image = np.array(pil_image) 
        return open_cv_image

def opencv2pil(numpy_image):
        im_pil = Image.fromarray(numpy_image)
        return im_pil

def pyrdown2D(signal, desired_filter):
    img_pil = signal.image_pil()
    img_npy = pil2opencv(img_pil)    

    filtered_decimated = desired_filter(img_npy)
    pil_filtered_decimated = opencv2pil(filtered_decimated)

    w_new, h_new = pil_filtered_decimated.size
    tensor_filt_decimated = to_tensor(pil_filtered_decimated)

    return ImageSignal(tensor_filt_decimated,
                        w_new, 
                        h_new,
                        channels=signal.channels,
                        sampling_scheme=signal.sampling_scheme,
                        batch_samples_perc=signal.batch_samples_perc,
                        attributes=signal.attributes)

def pyrup2d_opencv(image,num_times, dims_to_upscale):
    img_scale = image
    for level in range(num_times):
        img_scale = cv2.pyrUp(img_scale,dstsize=dims_to_upscale[level])

    img_scale = cv2.pyrUp(img_scale, dstsize=dims_to_upscale[-1])
    
    return img_scale

def pyrup2D_imagesignal(signal,num_times, dims_to_upscale):
    img_pil = signal.image_pil()
    img_npy = pil2opencv(img_pil)    

    scaled_up_image = pyrup2d_opencv(img_npy,num_times,dims_to_upscale)
    pil_scaled_up_image = opencv2pil(scaled_up_image)

    w_new, h_new = pil_scaled_up_image.size
    tensor_scaled_up_image = to_tensor(pil_scaled_up_image)

    return ImageSignal(tensor_scaled_up_image,
                        w_new,
                        h_new,
                        channels=signal.channels,
                        sampling_scheme=signal.sampling_scheme,
                        batch_samples_perc=signal.batch_samples_perc,
                        attributes=signal.attributes
                        )

def construct_pyramid(signal, num_levels, desired_filter = cv2.pyrDown):
    pyramid = [signal]
    for _ in range(num_levels-1):
        signal = pyrdown2D(signal, desired_filter)
        pyramid.append(signal)
    return pyramid

def construct_gaussian_tower(gaussian_pyramid):
    pyramid_dimensions = [signal_dims.dimensions() for signal_dims in gaussian_pyramid[:-1] ]
    gauss_tower=[gaussian_pyramid[0]]
    for level,signal in enumerate(gaussian_pyramid[1:]):
        dims_to_upscale = pyramid_dimensions[:(level+1)]
        dims_to_upscale.reverse()
        signal = pyrup2D_imagesignal(signal,level,dims_to_upscale)
        gauss_tower.append(signal)
    return gauss_tower

def construct_laplacian_tower(gaussian_tower):
    laplacian_tower = [upper_gauss - lower_gauss for upper_gauss, lower_gauss in zip(gaussian_tower[:-1], gaussian_tower[1:])]
    laplacian_tower.append(gaussian_tower[-1])
    return laplacian_tower

def construct_laplacian_pyramid(gaussian_pyramid):
    laplacian_pyramid = [upper_gauss - pyrup2D_imagesignal(lower_gauss,num_times=0,dims_to_upscale=[upper_gauss.dimensions()]) 
                            for upper_gauss, lower_gauss 
                            in zip(gaussian_pyramid[:-1], gaussian_pyramid[1:])]
    laplacian_pyramid.append(gaussian_pyramid[-1])
    return laplacian_pyramid

def create_MR_structure(img_signal, num_levels, filter, decimation = False):

    if filter=='none' and not decimation:
        return [img_signal]*num_levels

    if filter=='none' and decimation:
        return construct_pyramid(img_signal,num_levels, desired_filter=resize_half_image)

    gaussian_pyramid = construct_pyramid(img_signal,num_levels)

    if filter=='gauss' and decimation:
        return gaussian_pyramid

    if filter=='laplace' and decimation:
        laplacian_pyramid = construct_laplacian_pyramid(gaussian_pyramid)
        return laplacian_pyramid
    
    gaussian_tower = construct_gaussian_tower(gaussian_pyramid)

    if filter=='gauss' and not decimation:
        return gaussian_tower

    if filter=='laplace' and not decimation:
        laplacian_tower = construct_laplacian_tower(gaussian_tower)
        return laplacian_tower



    



    