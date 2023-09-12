import numpy as np


if __name__ == '__main__':
    res = 512
    x = np.linspace(-1, 1, res)
    signal = np.power(np.e, x) * (np.cos(2*np.pi*x) + np.sin(7*np.pi*x)) + np.cos(4*np.pi*x)/4 + np.sin(16*np.pi*x)/16
    np.save("data/arrays/sinusoidal.npy", signal)