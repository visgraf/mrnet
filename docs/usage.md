# Using MR-Net

You can use any component of the MR-Net framework individually in your project. However, we provide a complete framework for training signals in multiresolution and a convenient way of changing the necessary hyperparameters for various experiments. Examples are available in the directory [docs/examples](https://github.com/visgraf/mrnet/tree/main/docs/examples):

The hyperparameters are listed in a YAML file. This way, you can configure many experiments without changing your code.

## Hyperparameters

### Naming
- `project_name`: a name for a set of experiments; if logging into Weights and Biases, this will be the name of the project created.

#### Network
- `model`: the model subclass, M for M-Net; L for L-Net; S-Net, will be incorporated into this code later.
- `in_features`: the dimension of the input layer (ex: 2 for an image)
- `out_features`: the dimension of the output layer (ex: 3 for three color channels)
- `hidden_layers`: number of hidden layers (ex: 1)
- `hidden_features`: number of features in the hidden layers; should be a list with one value for each hidden layer (ex: [256]) or a list with a pair [input, output] for each hidden layer (ex: [128, 256]).
- `bias`: boolean that states whether to have a bias in the first layer.
- `max_stages`: maximum number of stages to be added to the network (ex: 3)

#### Frequency Initialization
- `omega_0`: a list with 1 number for each stage of the network (ex: [16]); the range of frequencies from where we sample frequencies to initialize the first layer of the network.
- `hidden_omega_0`: a list with 1 number for each stage of the network (ex: [16]); the range of frequencies from where we sample frequencies to initialize the hidden layers of the network.
- `period`: a number; if period $\gt$ 0, the first layer of each stage will be initialized with integer multiples of this period, and the network will be periodic; otherwise, we draw frequencies from a "real" (floating-point) interval.
- `superposition_w0`: if it is `False`, a frequency chosen in the initialization of a stage will not appear in the initialization of subsequent stages; it only works for periodic signals, where these frequencies are based on integers.

#### Sampling
- `domain`: a pair of numbers or a list of pairs of numbers (ex: [-1, 1] or [[-1, 1], [-2, 2]]) 
- `sampling_scheme`: the sampling scheme used for the data; should be one of the values in: [regular, poisson]; **regular** applies regular sampling inside the domain interval; **poisson** applies Poisson disk sampling inside the domain; 
- `decimation`: a boolean (ex: True). If **True**, the signal will be downsampled by a factor of 2 after filtering (for pyramids); if **False**, it will not (for towers).
- `filter`: the filter used to build a multiresolution structure (pyramid or tower); should be one of the values in [gauss, Laplace, none]
- `pmode`: determines how the signal borders are handled; should be one of the valid [values specified here](https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.pyramid_gaussian).
- `attributes`: at this moment, should be either  ['d0', 'd1']. d0 corresponds to the signal value, and d1 corresponds to the signal gradient.


#### Loss
- `loss_function`: one of the following, 'mse' - squared L2 norm of the signal value (d0) or 'hermite' linear combination of mse for the signal and gradient.
- `loss_weights`: for 'hermite' use {'d0': 1, 'd1': 0.0} 

##### Training
- `opt_method`: the optimizer class used for training; should be one of the values in: [Adam]
- `lr`: a float (ex: 0.0001) for the learning rate used in the optimization.
- `max_epochs_per_stage`: an integer (ex: 800) for the maximum number of epochs to train each network stage.
- `batch_size`: an integer or an expression (ex: 128 * 128) for the number of samples (coordinates) of the signal used in each batch.
- `loss_tol`: a float (ex: 1e-10); if the loss function reaches a value below `loss_tol`, the training of the current stage will be interrupted.
- `diff_tol`: a float (ex: 1e-7); if the difference between the values of the loss function in two successive epochs is lower than `diff_tol`, the training of the current stage will be interrupted.

##### Data
- `data_path`: the path to the image file or numpy array file.
- `nsamples`: a number (applies only for 1D signals).
- `width`: an integer value (ex: 128) representing the *width* of the image signal; if it is greater than zero, the image will be resized; otherwise, its original size will be preserved.
- `height`: an integer value (ex: 128) representing the *height* of the image signal; if it is greater than zero, the image will be resized; otherwise, its original size will be preserved.
- `channels`: an integer value (ex: 3) representing the number of channels in the signal; should match `out_features`. Use 0 to have it automatically computed in the examples.
- `color_space`: RGB; for valid values, see: [Pillow docs](https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes)

#### Etc
- `logger`: should be **wandb** to log results to Weights and Biases service or a path to a local directory to save the results locally. 
- `device`: device used for computation during training (ex: cuda).
- `eval_device`: device used for computation during inference for logging results (ex: cpu).
- `visualize_grad`: a boolean (ex: True) representing whether it should generate visualizations of the magnitude of the gradients of the signal.
- `extrapolate`: a pair of values representing an interval to visualize the learned signal (ex: [-2, 2]); in higher dimensions, you can specify a pair for each direction (ex: [[-2, 2], [-3, 3]] would be $[-2, 2] \times [-3, 3]$)
- `zoom`: a list of numbers (ex: [2, 4]); for each value $z$ in the list, a visualization of the signal with a zoom factor of $z\times$ will be generated.
- `zoom_filters`: a list of baseline filters for evaluation of the zoom quality; should have values from: `['linear', 'cubic', 'nearest']`