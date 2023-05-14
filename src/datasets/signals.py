import warnings
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from pathlib import Path
from .constants import SAMPLING_DICT, Sampling
from datasets.sampler import SamplerFactory, make_grid_coords
from scipy.ndimage import sobel
from torch.utils.data import BatchSampler
from scipy.interpolate import RegularGridInterpolator, interpn
from .utils import rotation_matrix


def make_mask(srcpath, mask_color):
    img = np.array(Image.open(srcpath))
    mask = img != mask_color
    path = Path(srcpath)
    path = path.parent.absolute().joinpath("mask.png")
    Image.fromarray(mask).save(path)
    return str(path)


class BaseSignal(Dataset):
    def __init__(self, data,
                 domain=[-1, 1],
                 attributes=[], 
                 sampling_scheme=Sampling.REGULAR,
                 batch_size=0,
                 YCbCr=False,
                 shuffle=True):
        
        self.data = data
        self.domain = domain
        self.attributes = attributes
        self.ycbcr = YCbCr
        self.data_attributes = {}
        if 'd1' in self.attributes:
            self.compute_derivatives()
        if isinstance(sampling_scheme, str):
            sampling_scheme = SAMPLING_DICT[sampling_scheme]
        self.sampler = SamplerFactory.init(sampling_scheme,
                                           self.data,
                                           self.domain,
                                           self.data_attributes,
                                           batch_size,
                                           shuffle)
    def compute_derivatives(self):
        dims = len(self.data.shape[1:])
        directions = []
        for d in range(1, dims + 1):
            directions.append(
                torch.from_numpy(sobel(self.data, d, mode='wrap'))
                )
        # channels x N x dims
        self.data_attributes = {'d1': torch.stack(directions, dim=-1)}

    def size(self):
        return self.data.shape
    
    @property
    def shape(self):
        return self.size()
    
    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx):
        return self.sampler[idx]
    
    @property
    def batch_size(self):
        return self.sampler.batch_size
    
    def type_code(self):
        return 'B'
    
    def new_like(other, data, shuffle=None):
        if other.type_code() == '1D':
            Class = Signal1D
        elif other.type_code() == '2D':
            Class = ImageSignal
        elif other.type_code() == '3D':
            Class = VolumeSignal
        else:
            Class = BaseSignal
        if shuffle is None:
            shuffle = other.sampler.shuffle
        return Class(data, other.domain,
                    other.attributes, 
                    other.sampler.scheme(),
                    other.sampler.batch_size,
                    other.ycbcr,
                    shuffle)
        

class Signal1D(BaseSignal):
    def init_fromfile(filepath, 
                      domain=[-1, 1], 
                      attributes=[],
                      sampling_scheme=Sampling.REGULAR,
                      batch_size=-1):
        
        data = np.load(filepath)
        if isinstance(sampling_scheme, str):
            sampling_scheme = SAMPLING_DICT[sampling_scheme]

        return Signal1D(data.view(1, -1), domain, 
                        attributes, sampling_scheme, batch_size)
    
    def type_code(self):
        return "1D"


class ImageSignal(BaseSignal):

    def init_fromfile(imagepath, 
                      domain=[-1, 1],
                      channels=3,
                      sampling_scheme=Sampling.REGULAR,
                      width=None, height=None,
                      attributes=[], 
                      maskpath=None,
                      batch_size=0,
                      YCbCr=False):
        img = Image.open(imagepath)
        if channels == 1:
            img = img.convert('L')
        if channels == 3 and YCbCr:
            img = img.convert('YCbCr')

        if width is not None or height is not None:
            if height is None:
                height = img.height
            if width is None:
                width = img.width
            if width > img.width or height > img.height:
                warnings.warn(f"Resizing to a higher resolution ({width}x{height})", RuntimeWarning)
            img = img.resize((width, height))
        img_tensor = to_tensor(img)
        
        mask = to_tensor(Image.open(maskpath).resize((width, height))) if maskpath else None
        if isinstance(sampling_scheme, str):
            sampling_scheme = SAMPLING_DICT[sampling_scheme]

        return ImageSignal(img_tensor,
                            domain=domain,
                            sampling_scheme=sampling_scheme,
                            attributes=attributes,
                            batch_size=batch_size,
                            YCbCr=YCbCr)
                            #domain_mask=mask)

    def compute_derivatives(self):
        dims = len(self.data.shape[1:])
        directions = []
        for d in range(1, dims + 1):
            directions.append(
                torch.from_numpy(sobel(self.data, d, mode='wrap'))
                )
        # channels x N x dims
        if self.ycbcr:
            self.data_attributes = {'d1': torch.stack(directions, dim=-1)[0:1, ...]} #GAMBIARRA YCbCr
        else:
            self.data_attributes = {'d1': torch.stack(directions, dim=-1)}

    def image_pil(self):
        return to_pil_image(self.data)
    
    def type_code(self):
        return "2D"

    def __sub__(self,other):
        if self.domain != other.domain:
            raise NotImplementedError("Can only operate signals in same domain for now")
        data_self = self.image_t
        data_other = other.image_t
        subtract_data = data_self - data_other
        width,height = self.dimensions()
        return ImageSignal(subtract_data,
                            domain=self.domain,
                            sampling_scheme=self.sampling_scheme,
                            batch_samples_perc=self.batch_samples_perc,
                            attributes=self.attributes)
    
class VolumeSignal(BaseSignal):

    def init_fromfile(volumepath, 
                      domain=[-1, 1],
                      sampling_scheme=Sampling.REGULAR,
                      width=None, height=None,
                      attributes=[], channels=3,
                      batch_size=0,
                      maskpath=None):
        volume = np.load(volumepath).astype(np.float32)
        
        if channels == 1:
            volume = (0.2126 * volume[:, :, :, 0] 
                      + 0.7152 * volume[:, :, :, 1] 
                      + 0.0722 * volume[:, :, :, 2])
            volume = np.expand_dims(volume, axis=0)
        else:
            if volume.shape[-1] == 3:
                volume = volume.transpose((3, 0, 1, 2))

        if width is not None or height is not None:
            raise NotImplementedError("Can't resize volume at this moment")
        vol_tensor = torch.from_numpy(volume)
        # mask = to_tensor(Image.open(maskpath).resize((width, height))) if maskpath else None

        if isinstance(sampling_scheme, str):
            sampling_scheme = SAMPLING_DICT[sampling_scheme]
        mask = None
        print(vol_tensor.shape, 'VOLUME SHAPE')
        return VolumeSignal(vol_tensor,
                            domain=domain,
                            sampling_scheme=sampling_scheme,
                            attributes=attributes,
                            batch_size=batch_size)
                            #domain_mask=mask)

    def type_code(self):
        return "3D"
    
    def get_slices(self, dircodes):
        slices = []
        volume = self.data
        x_slice = 1
        y_slice = 2
        z_slice = 3
        for code in dircodes:
            if code == 'x':
                slices.append(volume[:, x_slice, :, :].permute((1, 2, 0)))
            if code == 'y':
                slices.append(volume[:, :, y_slice, :].permute((1, 2, 0)))
            if code == 'z':
                slices.append(volume[:, :, :, z_slice].permute((1, 2, 0)))
            if len(code) == 2:
                code_map = { 'x': [2, 0, 1], 'y': [0, 2, 1], 'z': [0, 1, 2] }
                dims = self.size()[0]
                res = self.size()[1]
                coords = make_grid_coords(res, *self.domain, dims - 1)
                newdim = 0.0 * torch.ones((len(coords), 1))
                domain = torch.cat([coords, newdim], dim=-1)
                domain = domain[:, code_map[code[0]]]
                rotation = rotation_matrix(code[1], np.pi/4)
                domain = torch.matmul(rotation, 
                                        domain.permute((1, 0))).permute((1, 0))
                
                # interpolate values
                points = tuple(np.linspace(*self.domain, res) 
                                for i in range(dims))

                values = []
                for c in range(len(self.data)):
                    values.append(
                        interpn(points, self.data[c].numpy(), domain.numpy())
                    )
                values = torch.from_numpy(np.array(values)).permute((1, 0))
                slices.append(
                    values.reshape((res, res, len(self.data))).float())
        
        return slices

    
# TODO make a procedural sampler
class Procedural3DSignal(Dataset):
    def __init__(self, procedure,
                 dims,
                 channels,
                 domain=[-1, 1],
                 attributes=[], 
                 sampling_scheme=Sampling.REGULAR,
                 batch_size=0,
                 YCbCr=False,
                 shuffle=True):
        
        self.procedure = procedure
        self.dims = dims
        self.channels = channels
        # TODO use shuffle
        self.domain = domain
        self.batch_size = batch_size
        self.attributes = attributes
        self.ycbcr = YCbCr
        self.data_attributes = {}
        # if 'd1' in self.attributes:
        #     self.compute_derivatives()
        # if isinstance(sampling_scheme, str):
        #     sampling_scheme = SAMPLING_DICT[sampling_scheme]
        # self.sampler = SamplerFactory.init(sampling_scheme,
        #                                    self.data,
        #                                    self.domain,
        #                                    self.data_attributes,
        #                                    batch_size,
        #                                    shuffle)
        self.order = 0
        slice_size = dims[0] * dims[1]
        self.batches_per_slice = slice_size // batch_size
        if slice_size % batch_size != 0:
            self.batches_per_slice += 1

    def size(self):
        return (self.channels, *self.dims)
    
    @property
    def shape(self):
        return self.size()
    
    def __len__(self):
        # return len(self.sampler)
        return self.batches_per_slice * self.dims[self.order]

    def __getitem__(self, idx):
        try:
            coords = next(self.coords_sampler)
        except: #which?
            self.newdim = torch.linspace(self.domain[0], 
                                        self.domain[1], 
                                        self.dims[self.order])
            # TODO: deal with different sizes in each direction and ORDER
            coords = make_grid_coords(self.dims[0], 
                                      *self.domain, 
                                      len(self.dims)-1)
            value = self.newdim[idx // self.batches_per_slice]
            coords = torch.cat([coords, 
                                value * torch.ones((len(self.newdim)**2, 1))],
                                dim=-1)
            self.coords_sampler = iter(BatchSampler(coords, 
                                               self.batch_size, 
                                               drop_last=False))
            coords = next(self.coords_sampler)
        coords = torch.stack(coords)
        in_dict = {'coords': coords}
        out_dict = {'d0': self.procedure(coords)}
        # if 'd1' in self.attributes.keys():
        # #     # lazy or not?
        #     out_dict['d1'] = self.d1[sel_idxs]
        return {'c0': (in_dict, out_dict)}
    
    def type_code(self):
        return 'P'
    
    def new_like(other, data, shuffle=None):
        raise NotImplementedError("Procedural Signal does not support this operation")
    
    def get_slices(self, dircodes):
        valid_codes = ['x', 'y', 'z', 'xy', 'xz', 'yz']
        code_map = {
            'x': [2, 0, 1],
            'y': [0, 2, 1],
            'z': [0, 1, 2]
        }
        coords = make_grid_coords(self.dims[0], 
                                  *self.domain, 
                                  len(self.dims) - 1)
        newdim = 0.0 * torch.ones((len(coords), 1))
        domain_slice = torch.cat([coords, newdim], dim=-1)
        slices = []
        res = self.dims[0]
        for code in dircodes:
            if code not in valid_codes:
                raise ValueError(
                    "Direction codes should be in [x, y, z, xy, xz, yz]")
            domain = domain_slice[:, code_map[code[0]]]
            if len(code) == 2:
                # todo, check hot to do it
                rotation = rotation_matrix(code[1], np.pi/4)
                domain = torch.matmul(rotation, 
                                      domain.permute((1, 0))).permute((1, 0))
            slices.append(
                self.procedure(domain).reshape(res, res, self.channels))

        return slices
