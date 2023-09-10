# MR-Net - Multiresolution Sinusoidal Neural Networks

MR-Net is a framework that implements the family of neural networks described in [[1]](#1), and the components for training multi-stage architectures for multiresolution signal representation

This framework has 3 big components:

### Networks

Here, you will find the implementations of M-Net, L-Net, and a modified version of Siren [[2]](#2), as we build on top of their sinusoidal layer. You can create instances of the MR-Net subclasses directly importing them from the `mrnet` module.

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
    pip install git+https://github.com/visgraf/mrnet.git@public
```

#### Optional

If you want to log your results to [Weights and Biases](https://wandb.ai), you should also install the wandb package:

```
    pip install wandb
```

If you want to run the sample Jupyter notebooks locally, please, follow the [installation instruction from the Jupyter project page](https://jupyter.org/install).


## Testing

python tests/test_image.py

## Using MR-Net

You can find a sample notebook showing [how to use the MR-Net framework here](https://colab.research.google.com/drive/1Q88hgGa4Pvt-CsBjqI0dHs3wrEYJKMFo?usp=sharing).

You can use any component of the MR-Net framework individually in your project. However, we povide a complete framework for training signals in multiresolution and a convenient way of changing the necessary hyperparameters fo a variety of experiments. An example for image representation is available in the following notebook:

The hyperparameters are listed in an YAML file. This way, you can configure many experiments witout having to change your code.

## Hyperparameters

#### Naming
- `project_name`: a name for a set of experiments; if logging to Weights and Biases, this will be the name of the project created.

#### Network
- `model`: the model subclass; M for M-Net; L for L-Net; S-Net will be incorporated to this code later.
- `in_features`: dimension of the input layer (ex: 2 for an image)
- `out_features`: dimension of the output layer (ex: 3 for three color channels)
- `hidden_layers`: number of hidden layers (ex: 1)
- `hidden_features`: number of features in the hidden layers; should be a list with 1 value for each hidden layer (ex: [256]) or a list with a pair [input, output] for each hidden layer (ex: [128, 256]).
- `bias`: boolean that states whether to have a bias in the first layer or not.
- `max_stages`: maximum number of stages to be added to the network (ex: 3)

#### Frequency Initialization
- `omega_0`: a list with 1 number for each stage of the network (ex: [16]); the range of frequencies from where we sample frequencies to initialize the first layer of the network.
- `hidden_omega_0`: a list with 1 number for each stage of the network (ex: [16]); the range of frequencies from where we sample frequencies to initialize the hidden layers of the network.
- `period`: a number; if period $\gt$ 0, the first layer of each stage will be initialized with integer multiples of this period and the network will be periodic; otherwise, we draw frequencies from a "real" (floating-point) interval.
- `superposition_w0`: if it is `False`, a frequency chosen in the initialization of a stage will not appear in the initialization of subsequent stages; it only works for periodic signals, where this frequencies are based on integers.

#### Sampling
- `domain`: a pair of numbers or a list of pairs of numbers (ex: [-1, 1] or [[-1, 1], [-2, 2]]) 
- `sampling_scheme`: the sampling scheme used for the data; should be one of the values in: [regular, reflect, poisson, stratified]; **regular** applies regular sampling inside the domain interval; **reflect** doubles the domain interval, reflects the signal data and applies regular sampling in the extended signal; **poisson** applies Poisson disk sampling inside the domain; **stratified** (untested!) samples the critical points of the signal and some stochastic points using Poisson disck sampling.
- `decimation`: a boolean (ex: True). if **True**, the signal will be downsampled by a factor of 2 after filtering (for pyramids); if **False**, it will not (for towers).
- `filter`: the filter used to build a multiresolution structure (pyramid or tower); should be one of the values in [gauss, laplace, none]
- `pmode`: determines how the signal borders are handled; should be one of the valid [values specified here](https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.pyramid_gaussian).
- `attributes`: at this moment, should be either  ['d0', 'd1']


#### Loss (TODO)
- `loss_function`: 'hermite'
- `loss_weights`: {'d0': 1, 'd1': 0.0}

##### Training
- `opt_method`: the optimizer class used for training; should be one of the values in: [Adam]
- `lr`: a float (ex: 0.0001) for the learning rate used in the optimization.
- `max_epochs_per_stage`: an integer (ex: 800) for the maximum number of epochs to train each stage of the network.
- `batch_size`: an integer or an expression (ex: 128 * 128) for the number of samples (coordinates) of the signal used in each batch.
- `loss_tol`: a float (ex: 1e-10); if the loss function reaches a value below `loss_tol`, the training of the current stage will be interrupted.
- `diff_tol`: a float (ex: 1e-7); if the difference between the values of the loss function in two successive epochs is lower than `diff_tol`, the training of the current stage will be interrupted.

##### Image
- `image_name`: the path to the image file.
- `width`: an integer value (ex: 128) representing the *width* of the image signal; if it is greater than zero, the image will be resized; otherwise, its original size will be preserved.
- `height`: an integer value (ex: 128) representing the *height* of the image signal; if it is greater than zero, the image will be resized; otherwise, its original size will be preserved.
- `channels`: an integer value (ex: 3) representing the number of channels in the signal; should match `out_features`.
- `color_space`: RGB; for valid values, see: [Pillow docs](https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes)

#### Etc
- `logger`: should be **wandb** to log results to Weights and Biases service or a path to a local directory to save the results locally. 
- `device`: device used for computation during training (ex: cuda).
- `eval_device`: device used for computation during inference for logging results (ex: cpu).
- `visualize_grad`: a boolean (ex: True) representing wether it should generate visualizations of the magnitude of the gradients of the signal.
- `extrapolate`: a pair of values representing an interval to visualize the learned signal (ex: [-2, 2]); in higher dimensions, you can specify a pair for each direction (ex: [[-2, 2], [-3, 3]] would be $[-2, 2] \times [-3, 3]$)
- `zoom`: a list of number (ex: [2, 4]); for each value $z$ in the list, a visualization of the signal with a zoom factor of $z\times$ will be generated.
- `zoom_filters`: a list of baseline filters for evaluation of the zoom quality; should have values from: `['linear', 'cubic', 'nearest']`

## References

<a id="1">[1]</a> Hallison Paz, Daniel Perazzo, Tiago Novello, Guilherme Schardong, Luiz Schirmer, Vinícius da Silva, Daniel Yukimura, Fabio Chagas, Hélio Lopes, Luiz Velho. MR-Net: Multiresolution sinusoidal neural networks. Computers & Graphics, Volume 114, 2023, Pages 387-400.

<a id="2">[2]</a> Vincent Sitzmann, Julien N.P. Martel, Alexander W. Bergman, David B. Lindell, and Gordon Wetzstein. Implicit neural representations with peri- odic activation functions. In Proc. NeurIPS, 2020.


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
