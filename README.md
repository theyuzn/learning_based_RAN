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


## Execute

```sh
# Run docker
sudo docker run --mount type=bind,source=/home/yz/research/docker/li_sche_pytorch,target=/workspace -it --gpus --name li_sche_pyt ubuntu:jammy-20230308 bash
```

## Credit

## CHANGELOG
