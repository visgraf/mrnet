import torch
import numpy as np
import random
from .signal1d import Signal1D
from scipy.ndimage import gaussian_filter1d
from scipy import interpolate
from .constants import Sampling


def find_inflection_idx(signal):
    smoothed = gaussian_filter1d(signal.data, 100)
    d2 = np.gradient(np.gradient(smoothed))
    inflectionidx = np.where(np.diff(np.sign(d2)))[0]
    return set(inflectionidx)

def find_zeros_indx(signal):
    zerosidx = np.where(np.diff(np.sign(signal.data)))[0]
    return set(zerosidx)

def find_extrema_idx(signal):
    smoothed = signal.data #gaussian_filter1d(signal.data, 100)
    d = np.gradient(smoothed)
    inflectionidx = np.where(np.diff(np.sign(d)))[0]
    return list(set(inflectionidx))


def add_stochastic_samples(signal, attributes, hyper):
    nsamples = hyper['stochastic_samples']
    minratio = 0.25 #TODO: could be a hyperparameter
    X = signal.timepoints
    xregular = np.linspace(X[0], X[-1], nsamples)
    # Y = signal.data
    # fsignal = interpolate.interp1d(X, Y, kind='cubic')
    noiseratio = (1.0 - minratio) / 2
    b = noiseratio * (xregular[1] - xregular[0])
    a = -b
    xnoise = (b - a) * np.random.random_sample(nsamples) + a
    # keep values inside interpolation interval
    xnoise[0], xnoise[-1] = abs(xnoise[0]), -abs(xnoise[-1])

    xnew = xregular + xnoise
    xtensor = torch.cat([attributes['x'], 
                            torch.tensor(xnew, dtype=torch.float32)])
    # We'll use sortidx to sort attributes based on X order
    xtensor, sortidx = torch.sort(xtensor)
    attributes['x'] = xtensor
   
    derivatives = {'d0': signal.data, 'd1':signal.d1, 'd2':signal.d2}
    for dn, signal_dn in derivatives.items():
        fdn = interpolate.interp1d(X, signal_dn, kind='cubic')
        stoch_dn = torch.tensor(fdn(xnew), dtype=torch.float32)
        cattensor = torch.cat([attributes[f'{dn}'], stoch_dn])
        attributes[f'{dn}'] = cattensor[sortidx]
        if hyper.get(f'stochastic_{dn}', False):
            stoch_dn_mask = torch.ones_like(stoch_dn, dtype=torch.bool)
        else:
            stoch_dn_mask = torch.zeros_like(stoch_dn, dtype=torch.bool)
        cattensor = torch.cat([attributes[f'{dn}_mask'], stoch_dn_mask])
        attributes[f'{dn}_mask'] = cattensor[sortidx]
   

def sample_signal(signal: Signal1D, hyper: dict) -> Signal1D:
    samplesidx = set()
    # always keep the endpoints
    samplesidx.add(0)
    samplesidx.add(len(signal) - 1)
    if signal._useattributes:
        d0_mask = torch.zeros_like(signal.data, dtype=torch.bool)
        d1_mask = torch.zeros_like(signal.d1, dtype=torch.bool)
        d2_mask = torch.zeros_like(signal.d2, dtype=torch.bool)
        # use all data on endpoints
        d0_mask[[0, -1]], d1_mask[[0, -1]], d2_mask[[0, -1]] = True, True, True
    
    sampling_map = {'zeros': find_zeros_indx, 'extrema': find_extrema_idx}
    for key, searchfunc in sampling_map.items():
        key_idx = searchfunc(signal)
        idx = sorted(key_idx)
        use_d0 = hyper.get(f'{key}_d0', False)
        use_d1 = hyper.get(f'{key}_d1', False)
        use_d2 = hyper.get(f'{key}_d2', False)
        if use_d0:
            d0_mask[sorted(idx)] = True
        if use_d1:
            d1_mask[sorted(idx)] = True
        if use_d2:
            d2_mask[sorted(idx)] = True
    
        if use_d0 or use_d1 or use_d2:
            samplesidx = samplesidx.union(key_idx)
    
    n_remaing_idx = hyper.get('random_samples', 0) 
    if n_remaing_idx > 0:
        allidx = set(range(len(signal)))
        unusedidx = allidx.difference(samplesidx)
        remainingidx = random.sample(tuple(unusedidx), n_remaing_idx)
        samplesidx = samplesidx.union(remainingidx)
        d0_mask[remainingidx] = True
        if hyper.get('remaining_d1', False):
            d1_mask[remainingidx] = True
        if hyper.get('remaining_d2', False):
            d2_mask[remainingidx] = True

    # get data from idx
    idx = sorted(samplesidx)
    # timepoints = signal.timepoints[idx]
    # data = signal.data[idx]
    data_map = {'x': signal.timepoints[idx],
                'd0': signal.data[idx],
                'd0_mask': d0_mask[idx], 
                'd1': signal.d1[idx],
                'd2': signal.d2[idx],
                'd1_mask': d1_mask[idx],
                'd2_mask': d2_mask[idx]}

    n_stochastic = hyper.get('stochastic_samples', 0)
    if n_stochastic > 0:
        add_stochastic_samples(signal, data_map, hyper)
    
    return Signal1D(-1, 
                    data=data_map['d0'],
                    timepoints=data_map['x'],
                    sampling_scheme=Sampling.MAX_MIN,
                    attributes=data_map)

def sample_max_min(signal: Signal1D, 
                    include_inflection=False,
                    include_zeros=False):

    samplesidx = set()
    # always keep the endpoints
    samplesidx.add(0)
    samplesidx.add(signal.num_samples() - 1)

    samplesidx = samplesidx.union(find_extrema_idx(signal))
    
    if include_inflection:
        samplesidx = samplesidx.union(find_inflection_idx(signal))
        
    if include_zeros:
        samplesidx = samplesidx.union(find_zeros_indx(signal))
    
    indices = sorted(samplesidx)
    timepoints = signal.timepoints[indices]
    data = signal.data[indices]

    attributes = None
    if signal._useattributes:
        d1 = signal.grad[indices]
        d2 = signal.d2[indices]
        attributes = {'d1': d1, 'd2': d2}
    
    return Signal1D(len(timepoints), 
                    data=data,
                    timepoints=timepoints,
                    sampling_scheme=Sampling.MAX_MIN,
                    attributes=attributes)

def sample_random_shake(signal: Signal1D, n_samples, minratio):
    if signal.sampling_scheme != Sampling.UNIFORM:
        raise ValueError("Can't re-sample non-uniform signals")
    
    X = signal.timepoints
    Y = signal.data
    fsignal = interpolate.interp1d(X, Y, kind='cubic')

    noiseratio = (1.0 - minratio) / 2
    xregular = np.linspace(X[0], X[-1], n_samples)
    b = noiseratio * (xregular[1] - xregular[0])
    a = -b
    xnoise = (b - a) * np.random.random_sample(n_samples) + a
    xnoise[0] = xnoise[-1] = 0.0

    xnew = xregular + xnoise
    ynew = fsignal(xnew)
    return Signal1D(n_samples, 
                    data=torch.tensor(ynew, dtype=torch.float32), 
                    timepoints=torch.tensor(xnew, dtype=torch.float32), 
                    sampling_scheme=Sampling.RANDOM_SHAKE)
