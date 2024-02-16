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