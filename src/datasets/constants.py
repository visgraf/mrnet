from enum import Enum, auto


class Sampling(Enum):
    UNIFORM = auto()
    MAX_MIN = auto()
    RANDOM_SHAKE = auto()
    POISSON_DISC = auto()

SAMPLING_DICT = {
    'uniform': Sampling.UNIFORM,
    'max_min': Sampling.MAX_MIN,
    'poisson': Sampling.POISSON_DISC
}