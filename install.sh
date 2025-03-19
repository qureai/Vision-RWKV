conda create -n vrwkv python=3.10 -y
conda activate vrwkv
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install mmcv-full
pip install ninja
pip install mmcls
pip install timm==0.6.12
pip install opencv-python termcolor yacs pyyaml scipy