---
hide:
  - navigation
  # - toc
---

# MR-Net

MR-Net is a framework that implements the family of neural networks described in [[1]](#1), and the components for training multi-stage architectures for multiresolution signal representation


## Installation Instructions

MRNet was tested with Python3.9 and Python3.11.

#### Environment

We suggest using either Venv or Anaconda Environments.

Python Venv:
```bash
    python -m venv venv
    venv/Scripts/activate
```

Anaconda:
```bash
    conda create -n mrnet python=3.11
    conda activate mrnet
```

#### Dependencies

On Windows systems:
```bash
    pip install -r requirements.txt
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

On Linux or Mac OS systems:
```bash
    pip install -r requirements.txt
    pip install torch torchvision torchaudio
```

After installing the dependencies, install the MR-Net package as follows:

```bash
    pip install git+https://github.com/visgraf/mrnet.git
```
OBS: Note that this command installs MR-Net in your environment (Venv or Conda), therefore the files in the ./mrnet/ folder serve only as a reference for the package.

#### Optional

If you want to log your results to [Weights and Biases](https://wandb.ai), you should also install the wandb package:

```bash
    pip install wandb
```

If you want to run the sample Jupyter notebooks locally, please, follow the [installation instruction from the Jupyter project page](https://jupyter.org/install).


## Testing MR-Net

Testing MR-Net involves training a signal and inference of the model.

### Training

Training MR-Net to reconstruct a signal (1D, or image) consists of providing input data samples (i.e., pairs of coordinate and attribute values) to produce a continuous model representation given by the MR-Net.

Examples of Python code to train signals with MR-Net can be found in the directory ./docs/examples. These examples use configuration files in ./docs/configs.

##### 1D signal example:
```
python docs\examples\train_signal1d.py
```
##### Image example:
```
python docs/examples/train_image.py
```
The trained model is stored locally in the directory ./runs/logs/{model-name-dir}. 

### Inference

After training with MR-Net to create a signal representation, the model can be used to reconstruct the signal by evaluating the network at any continuous location in space and scale.

The jupyter notebook in ./docs/examples/ exemplifies the evaluation capabilities of MR-Net image model.

##### Evaluation example:
```
jupyter notebook
   (open docs\examples\eval-net.ipynb)
```
The notebook asks for the location of the configuration file and the trained model.



## References

<a id="1">[1]</a> Hallison Paz, Daniel Perazzo, Tiago Novello, Guilherme Schardong, Luiz Schirmer, Vinícius da Silva, Daniel Yukimura, Fabio Chagas, Hélio Lopes, Luiz Velho. MR-Net: Multiresolution sinusoidal neural networks. Computers & Graphics, Volume 114, 2023, Pages 387-400.

<a id="2">[2]</a> Vincent Sitzmann, Julien N.P. Martel, Alexander W. Bergman, David B. Lindell, and Gordon Wetzstein. Implicit neural representations with periodic activation functions. In Proc. NeurIPS, 2020.


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
