# Multiresolution Neural Networks for Imaging

## Installation Instructions

cd [parent_dir]  
git clone https://github.com/visgraf/mrimaging.git   
cd mrimaging  
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