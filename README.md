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

* Build docker file
```sh=
cd where_you_clone_this_project/dockerfile
sudo docker build -t li_sche_pyt . --no-cache
```


## Execute

```sh
cd where_you_clone_this_project
# Run docker
sudo docker run --mount type=bind,source="$(pwd)"/,target=/workspace -it --gpus all --name li_sche_pyt li_sche_pyt bash
```

## Credit

## CHANGELOG
