# MR-Net - Multiresolution Sinusoidal Neural Networks

## Installation Instructions

pip install git+https://github.com/visgraf/mrnet.git@dev

---
**or**

cd [parent_dir]  
git clone https://github.com/visgraf/mrnet.git   
cd mrimg
mkdir models  

mkdir venv  
python -m venv venv  
pip freeze  
python -m pip install --upgrade pip  

venv/Scripts/activate  
pip install -r .\requirements.txt  

pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu113 

## Testing

code .  
[open src/*.ipynb]  
[select venv kernel]  
[run all cells]  
