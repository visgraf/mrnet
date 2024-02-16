This framework has three major components:

### Networks

Here, you will find the implementations of M-Net, L-Net, and a modified version of Siren [[2]](#2) as we build on top of their sinusoidal layer. You can create instances of the MR-Net subclasses directly importing them from the `mrnet` module.

### Datasets

In the module `signals,` you will find the classes `Signal1D` and `ImageSignal.` They are subclasses of PyTorch Dataset and encapsulate the data fed to the network for training. In the module `procedural`, there are helper functions to generate procedural signals such as Perlin noise adapted to our datasets classes.

If you want to make your custom dataset, you could subclass `BaseSignal` or use the mentioned classes as a template to guide you.

The other modules contain helper functions to sample the signals, build the multiresolution structure, or make common operations such as color space transform.

### Training

In the module `trainer`, you will find the `MRTrainer` class, which encapsulates all the PyTorch code necessary for training a model for a certain amount of epochs, and manages the multiresolution structure for escalonated training of the networks. Without an `MRTrainer` instance, you must define when to add new stages to the network and how to train each stage. 