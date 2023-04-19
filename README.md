# li_sche_pytorch

## File Structure
```sh=
----li_sche_pytorch/
	|----README.md							# This file
	|----src/								# The simulation folder
		|----gNB/							# The gNB node
		|	|----pf_scheduler/				# The scheduler using PF algorithm
		|	|----rr_scheduler/				# The scheduler using RR algorithm
		|	|----lightweight_scheduler/		# The scheduler using DQN
		|----ue/							# The UE node
```

## How to build
* Install cuda driver for docker `Only the first time`
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
```sh
cd where_you_clone_this_project
# Run docker
sudo docker run --mount type=bind,source="$(pwd)"/,target=/workspace -it --gpus all --name li_sche_pyt li_sche_pyt bash

# After access container, try cuda driver
nvidia-smi
```
* `--gpus` to enable docker to access cuda driver
* `--mount` to enable docker to access the file with type bind
 
## Credit

## CHANGELOG
