import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor
# from .imagesignal import ImageSignal
from .signals import ImageSignal, BaseSignal
import cv2
import numpy as np
from PIL import Image

import scipy.ndimage as sig
from skimage.transform import pyramid_gaussian, pyramid_laplacian, pyramid_expand


def resize_half_image(numpy_image):
    dims = numpy_image.shape
    resized_img = cv2.resize(numpy_image, (dims[0]//2, dims[1]//2),
                              interpolation = cv2.INTER_AREA)
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
                        domain=signal.domain,
                        batch_size=signal.batch_size,
                        attributes=signal.attributes)

def pyrup2d_opencv(image,num_times, dims_to_upscale):
    img_scale = image
    for level in range(num_times):
        img_scale = cv2.pyrUp(img_scale,dstsize=dims_to_upscale[level])

    img_scale = cv2.pyrUp(img_scale, dstsize=dims_to_upscale[-1])
    
    return img_scale

def pyrup2D_imagesignal(signal, num_times, dims_to_upscale):
    img_pil = signal.image_pil()
    img_npy = pil2opencv(img_pil)    

    scaled_up_image = pyrup2d_opencv(img_npy,num_times, dims_to_upscale)
    pil_scaled_up_image = opencv2pil(scaled_up_image)

    w_new, h_new = pil_scaled_up_image.size
    tensor_scaled_up_image = to_tensor(pil_scaled_up_image)

    return ImageSignal(tensor_scaled_up_image,
                        w_new,
                        h_new,
                        channels=signal.channels,
                        domain=signal.domain,
                        sampling_scheme=signal.sampling_scheme,
                        batch_size=signal.batch_size,
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
        signal = pyrup2D_imagesignal(signal, level, dims_to_upscale)
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


VALID_FILTERS = {
    'gauss': pyramid_gaussian,
    'laplace': pyramid_laplacian,
}

def create_MR_structure(signal, num_levels, filter_name, 
                                decimation, mode='wrap', sigma=2/3,
                                channel_axis=0):
    if filter_name == 'none':
        if decimation:
            # it does not make sense on regular sampling; should re-sample
            raise NotImplementedError(f"Invalid for now: filter {filter} + decimation {decimation}")
        return [signal] * num_levels
    else:
        pyramid_filter = VALID_FILTERS[filter_name]
        
        signal_data = signal.data.numpy()
        pyramid = pyramid_filter(signal_data, 
                                num_levels-1,
                                sigma=sigma, mode=mode,
                                channel_axis=channel_axis)
        if decimation:
            return [BaseSignal.new_like(signal, torch.from_numpy(data)) 
                                    for data in pyramid]
        mrstack = []
        for i, sdata in enumerate(pyramid):
            current = sdata
            for j in range(i):
                current = pyramid_expand(current, 
                            sigma=sigma, 
                            mode=mode, 
                            channel_axis=channel_axis)
            mrstack.append(BaseSignal.new_like(signal, torch.from_numpy(current)))
        return mrstack



    



    