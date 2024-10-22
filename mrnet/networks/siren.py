import torch
import torch.nn.functional as F
import numpy as np

from torch import nn
from typing import Sequence
from collections import OrderedDict

RANDOM_SEED = 777

def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30, period=0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.period = period

        self.init_weights()

    def init_weights(self):
        if self.period != 0:
            self.init_periodic_weights()
        else:
            with torch.no_grad():
                if self.is_first:
                    self.linear.weight.uniform_(-1, 1) #* self.omega_0
                else:
                    self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                                np.sqrt(6 / self.in_features) / self.omega_0)
                
    def init_periodic_weights(self, used_weights=[]):
        # don't need to choose the origin
        if self.in_features == 2:   
            discarded_freqs = set([(0, 0)])
        else:
            discarded_freqs = set()
        discarded_freqs = discarded_freqs.union(set(used_weights))

        if isinstance(self.period, Sequence):
            # TODO: make it work to dimension > 2
            aspect_ratio = self.period[0] / self.period[1]
        else:
            aspect_ratio = 1

        with torch.no_grad():
            if self.is_first:
                rng = np.random.default_rng(RANDOM_SEED)
                if self.in_features == 2:
                    possible_frequencies = cartesian_product(
                        np.arange(0, self.omega_0 + 1), 
                        np.arange(int(-self.omega_0 / aspect_ratio), 
                                  int(self.omega_0 / aspect_ratio) + 1))
                else:
                    possible_frequencies = cartesian_product(
                        *(self.in_features * [np.array(range(-self.omega_0,
                                                            self.omega_0 + 1))])
                )
                if discarded_freqs:
                    possible_frequencies = np.array(list(
                        set(tuple(map(tuple, possible_frequencies))) - set(used_weights)
                    ))
                chosen_frequencies = torch.from_numpy(
                    rng.choice(possible_frequencies, self.out_features, False)
                )

                self.linear.weight = nn.Parameter(
                    chosen_frequencies.float() * 2 * torch.pi 
                    / torch.tensor(self.period))
                # first layer will not be updated during training
                self.linear.weight.requires_grad = False

    def forward(self, input):
        if self.period != 0 and self.is_first:
            x = self.linear(input)
        else:
            x = self.omega_0 * self.linear(input)
        return torch.sin(x)

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate



class Siren(nn.Module):
    """
    This SIREN version comes from:
    https://colab.research.google.com/github/vsitzmann/siren/blob/master/explore_siren.ipynb
    """
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, first_omega_0, hidden_omega_0,
                  bias=True, outermost_linear=False, superposition_w0=True):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, bias=bias,
                                  is_first=True, omega_0=first_omega_0))

        self.n_layers = hidden_layers + 1
        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def reset_weights(self):
        def reset_sinelayer(m):
            if isinstance(m, SineLayer):
                m.init_weights()
        self.apply(reset_sinelayer)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations


if __name__ == '__main__':
    a = SineLayer(2, 8, bias=True,
                 is_first=True, omega_0=30)
    print(a.linear.weight.shape, a.linear.weight.dtype)
    print(a.linear.weight)