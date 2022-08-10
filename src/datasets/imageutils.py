from decimal import InvalidOperation
import torch
import torch.nn.functional as F
from .imagesignal import ImageSignal
from .constants import Sampling

# In the future change this to use OpenCV

def convsignal2d(signal, kernel, padding_mode='reflect'):
    # padding is always 'same'

    if padding_mode == 'zeros':
        filtered = F.conv2d(signal.data.view(1, signal.channels, -1), 
                                kernel.view(1, signal.channels, -1), 
                                padding='same').view(-1)
    else:
        raise NotImplementedError("Only zeros padding for now")
        # kernel_size = kernel.shape
        # dilation = (1,)
        # reversed_padding_repeated_twice = [0, 0] * len(kernel_size)
        
        # for d, k, i in zip(dilation, kernel_size,
        #                     range(len(kernel_size) - 1, -1, -1)):
        #     total_padding = d * (k - 1)
        #     left_pad = total_padding // 2
        #     reversed_padding_repeated_twice[2 * i] = left_pad
        #     reversed_padding_repeated_twice[2 * i + 1] = (
        #         total_padding - left_pad)

        # filtered = F.conv1d(F.pad(signal.data.view(1, signal.channels, -1), 
        #                 reversed_padding_repeated_twice, mode='reflect'),
        #             kernel.view(1, signal.channels, -1))


def gaussian_kernel(size=5, device=torch.device('cpu'), channels=1):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    kernel = kernel.to(device)
    return kernel

def box_kernel2D(size=5, device=torch.device('cpu'), channels=1):
    kernel = torch.ones((size, size)) / size**2
    kernel = kernel.repeat(channels, 1, 1, 1)
    kernel = kernel.to(device)
    return kernel

def downsample(x):
    return x[:, :, ::2, ::2]

def downsample(signal: ImageSignal, factor=2):
    if signal.sampling_scheme == Sampling.UNIFORM:
        return ImageSignal(
            signal.data[::factor, ::factor, :],
            signal._width // 2 + 1,
            signal._height // 2 + 1,
            signal.coordinates[::factor, ::factor, :]
        )

    raise(InvalidOperation("Can't decimate a non uniformly sampled signal"))


def conv_gauss(img, kernel):
    n = kernel.shape[-1] - 1
    nlt = n // 2
    nrb = n - nlt
    img = torch.nn.functional.pad(img.unsqueeze(0), 
                                (nlt, nrb, nlt, nrb), mode='reflect')
    out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
    return out

def pyrdown2D(signal, kernel, decimate=True):    
    filtered = conv_gauss(signal.image_tensor(), kernel).squeeze(0)
    w, h = signal.dimensions()
    if decimate:
        return ImageSignal(torch.flatten(filtered[:, ::2, ::2]),
                            w // 2 + 1, h // 2 + 1,
                            None,
                            signal.channels,
                            useattributes=signal._useattributes)

    return ImageSignal(torch.flatten(filtered),
                        w, h,
                        signal.coordinates,
                        signal.channels,
                        useattributes=signal._useattributes)

def gaussian_pyramid2D(signal, levels, 
                        kernel=None, padding_mode='reflect'):
    if kernel is None:
        kernel = gaussian_kernel()
    pyramid = [signal]
    for s in range(levels-1):
        signal = pyrdown2D(signal, kernel)
        pyramid.append(signal)
    return pyramid

def gaussian_tower2D(signal, levels, kernel, padding_mode='reflect'):
    '''Only works with box filter for now'''
    
    tower = [signal]
    base_size = kernel.shape[-1]
    for s in range(levels-1):
        kernel = box_kernel2D(base_size * 2**s)
        signal = pyrdown2D(signal, kernel, decimate=False)
        tower.append(signal)
    return tower