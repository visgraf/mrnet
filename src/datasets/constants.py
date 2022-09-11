from enum import Enum, auto


class Sampling(Enum):
    UNIFORM = auto()
    POISSON_DISC = auto()

SAMPLING_DICT = {
    'uniform': Sampling.UNIFORM,
    'poisson': Sampling.POISSON_DISC
}