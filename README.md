# MR-Net - Multiresolution Sinusoidal Neural Networks

## Installation Instructions

MRNet was tested on version of Python $\ge$ 3.9.

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

