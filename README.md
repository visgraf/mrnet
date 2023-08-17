# MR-Net - Multiresolution Sinusoidal Neural Networks

MR-Net is a framework that implements the family of neural networks described in XXXX, and the components for training multi-stage architectures for multiresolution signal representation

This framework has 3 big components:

### Networks

Here, you will find the implementations of M-Net, L-Net, and a modified version of Siren [XXXX], as we build on top of their sinusoidal layer. You can create instances of the MR-Net subclasses directly importing them from the `mrnet` module.

### Datasets

In the module `signals` you will find the classes `Signal1D`, `ImageSignal`, and `VolumeSignal`. They are subclasses of PyTorch Dataset and encapsulate the data fed to the nework for training. In the module `procedural` there are helper functions to generate procedural signals such as Perlin noises adapted to our datasets classes.

If you want to make your custom dataset, you could subclass `BaseSignal` o just use the mentioned classes as a template to guide you.

The other modules contain helper functions to sample the signals, build the multiresolution structure, or make commom operations such as color space transform.

### Training

In the module `trainer`, you will find the `MRTrainer` class, which encapsulates all the PyTorch code necessary for training a model for a certain amount of epochs, and manages the multiresolution structure for escalonated training of the networks. Without a `MRTrainer` instance, you will need to define when to add new stages to the network and how to train each stage. 

## Installation Instructions

MRNet was tested with Python3.9 and Python3.11.

#### Dependencies

On windows systems:
```
    python -m venv venv
    venv/Scripts/activate
    pip install -r requirements.txt
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

On Linux or Mac OS systems:
```
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements_unix.txt
    pip install torch torchvision torchaudio
```

After installing the dependencies, install the MR-Net package as follows:

```
    pip install git+https://github.com/visgraf/mrnet.git@dev
```

## Testing

python tests/test_image.py

## Using MR-Net

You can use any component of the MR-Net framework individually in your project. However, we povide a complete framework for training signals in multiresolution and a convenient way of changing the necessary hyperparameters fo a variety of experiments. An example for image representation is available in the following notebook:


The hyperparameters are listed in an YAML file. This way, you can configure many experiments witout having to change your code.

## Hyperparameters

#### Naming
- project_name: a name for a set of experiments; if logging to Weights and Biases, this will be the name of the project created.

#### Network
- model: the model subclass; M for M-Net; L for L-Net; S-Net will be incorporated to this code later.
- in_features: dimension of the input layer (ex: 2 for an image)
- out_features: dimension of the output layer (ex: 3 for three color channels)
- hidden_layers: number off hidden layers (ex: 1)
- hidden_features: number of features in the hidden layers; should be a list with 1 value for each hidden layer (ex: [256]) or a list with a pair [input, output] for each hidden layer (ex: [128, 256]).
- bias: boolean that states whether to have a bias in the first layer or not.
- max_stages: maximum number of stages to be added to the network (ex: 3)

# Frequency Initialization
omega_0: a list with 1 number for each stage of the network (ex: [16]); the range of frequencies from where we sample frequencies to initialize the first layer of the network.
hidden_omega_0: a list with 1 number for each stage of the network (ex: [16]); the range of frequencies from where we sample frequencies to initialize the hidden layers of the network.
period: a number; if period $\gt$ 0, the first layer of each stage will be initialized with integer multiples of this period and the network will be periodic; otherwise, we draw frequencies from a "real" (floating-point) interval.
superposition_w0: if it is `False`, a frequency chosen in the initialization of a stage will not appear in the initialization of subsequent stages; it only works for periodic signals, where this frequencies are based on integers.

##### Signal

pmode: "wrap"
domain: [-1, 1]

# Sampling
sampling_scheme: regular
decimation: True
filter: gauss # vary between none, laplace and gauss
attributes: ['d0', 'd1']

# Loss
loss_function: 'hermite'
loss_weights: {'d0': 1, 'd1': 0.0}

# Training
opt_method: Adam
lr: 0.0001
loss_tol: 0.00000000001
diff_tol: 0.0000001
max_epochs_per_stage: 800
batch_size: 128 * 128

# Image
image_name: kodak512/girl_with_painted_face.png
width: 128
height: 128
channels: 3
#see: https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
color_space: RGB

# Computation (Only vary between 'cpu' and 'cuda')
device: cpu
eval_device: cpu

# Etc
save_format: 'general'
visualize_grad: True
extrapolate: [-2, 2]
zoom: [2, 4]
zoom_filters: ['linear', 'cubic', 'nearest']

<!-- positive_freqs: False -->



## Citing

If you use this repository in your research, consider citing it using the following Bibtex entry:

```
@article{PAZ2023387,
title = {MR-Net: Multiresolution sinusoidal neural networks},
journal = {Computers & Graphics},
volume = {114},
pages = {387-400},
year = {2023},
issn = {0097-8493},
doi = {https://doi.org/10.1016/j.cag.2023.05.014},
url = {https://www.sciencedirect.com/science/article/pii/S0097849323000699},
author = {Hallison Paz and Daniel Perazzo and Tiago Novello and Guilherme Schardong and Luiz Schirmer and Vinícius {da Silva} and Daniel Yukimura and Fabio Chagas and Hélio Lopes and Luiz Velho},
keywords = {Multiresolution, Level of detail, Neural networks, Imaging}
}
```

