#!/bin/bash
echo 'begin.......................'
sudo apt update 
sudo apt install make gcc -y
sudo apt install zlib1g
sudo apt install git
sudo apt install git-lfs
sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev


wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Linux-x86_64.sh
bash Miniconda3-py310_23.3.1-0-Linux-x86_64.sh

exit
##重新打开terminal实例

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

git clone https://github.com/MLRushOrg/ChatGLM2-6B.git
cd ChatGLM2-6B
pip install -r requirements.txt


exit
exit


nvidia驱动：********************************************************
'''
apt install nvidia-utils-525
check: nvidia-smi
'''

mkdir tmp
cd /tmp
https://www.nvidia.com/en-us/drivers/unix/
wget https://us.download.nvidia.com/tesla/525.105.17/NVIDIA-Linux-x86_64-525.105.17.run
sudo sh NVIDIA-Linux-x86_64-525.105.17.run
check: nvidia-smi 


cuda：*****************************************************************
###apt install nvidia-cuda-toolkit
https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_network
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb   ## 这个安装的是driver版本是530
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda
.bashrc
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
check: nvcc --version


cudnn:************************************************************
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2

sudo apt-get install libcudnn8=8.9.2.26-1+cuda12.1
sudo apt-get install libcudnn8-dev=8.9.2.26-1+cuda12.1
sudo apt-get install libcudnn8-samples=8.9.2.26-1+cuda12.1

check:
cp -r /usr/src/cudnn_samples_v8/ $HOME
cd  $HOME/cudnn_samples_v8/mnistCUDNN
make clean && make
./mnistCUDNN


git clone https://github.com/MLRushOrg/FastChat.git
cd FastChat
pip install setuptools_scm
pip install -e .
pip install transformers -U
pip install deepspeed -U

注意huggingface的cache，可能会让硬盘不够了。
export TRANSFORMERS_CACHE=/blabla/cache/

按照下面的方式打通ssh链接，才可以git clone huggingface上的文件
https://huggingface.co/docs/hub/security-git-ssh#generating-a-new-ssh-keypair
ssh-keygen -t ed25519 -C "liyongbao1988@gmail.com"
cat ~/.ssh/id_ed25519.pub
check:ssh -T git@hf.co

export GIT_LFS_SKIP_SMUDGE=1
git clone git@hf.co:baichuan-inc/baichuan-7B
cd baichuan-7B 
git lfs fetch -I "*.bin"

git clone用ssh的链接
https://discuss.huggingface.co/t/401-client-error-unauthorized-for-url/32466