# li_sche_pytorch
<p>The project is Lightweight Scheduler implemented in PyTorch</p>
<p>This research starts from 2023 and try to maintain in the future.</p>

# Clone
```sh
# To clone the submodel with the main repository
git clone --recurse-submodules
```
## File Structure
```sh=
----li_sche_pytorch/
	|----README.md							# This file
```

## Environment setup
* Install cuda driver for docker `Only the first time`
* !!!IF on the Linux with nvidia cuda GPU!!!
* !!! IF use MacOS, skip this step
```sh
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt update
sudo apt install -y nvidia-docker2
sudo systemctl restart docker
```

* Build docker file
```sh
cd where_you_clone_this_project/dockerfile
sudo docker build -t li_sche_pyt . --no-cache
```	

## Execute docker
* With GPU
```sh
cd [where_you_clone_this_project]

sudo docker run --mount type=bind,source="$(pwd)"/,target=/workdocker  -it --gpus all --name li_sche_pyt li_sche_pyt bash /bin/bash

nvidia-smi
```

* Without GPU (in MacOS)
```sh
cd [where_you_clone_this_project]

sudo docker run --mount type=bind,source="$(pwd)"/,target=/workdocker  -it  --name li_sche_pyt li_sche_pyt bash
```

## Build & Run
* In the first time you build this project, you need to install the pysctp module FIRST.
* Only need to do this one time.
```sh
cd src/li_sche/utils/pysctp  

sudo python3 setup.py install
python setup.py build
```
 
## Credit
1. Yu-Hsin Chuang

## CHANGELOG
