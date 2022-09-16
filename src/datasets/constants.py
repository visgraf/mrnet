from enum import Enum, auto


class Sampling(Enum):
    UNIFORM = auto()
    POISSON_DISC = auto()

SAMPLING_DICT = {
    'uniform': Sampling.UNIFORM,
    'poisson': Sampling.POISSON_DISC,
}

class Multiresolution(Enum):
    GAUSS_PYRAMID = auto()
    GAUSS_TOWER = auto()
    LAPLACE_PYRAMID = auto()
    LAPLACE_TOWER = auto()
    SIGNAL = auto()
    
MULTIRESOLUTION_DICT = {
    'gauss_pyramid': Multiresolution.GAUSS_PYRAMID,
    'gauss_tower': Multiresolution.GAUSS_TOWER,
    'laplace_pyramid': Multiresolution.LAPLACE_PYRAMID,
    'laplace_tower': Multiresolution.LAPLACE_TOWER,
    'signal': Multiresolution.SIGNAL
}
    