#!/bin/bash

# docker run --rm -it --runtime=nvidia --cap-add=SYS_PTRACE --security-opt="seccomp=unconfined" -v $HOME/coding/:/opt -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix rin/torch:cuda-10.1-cudnn7-devel-ubuntu18.04

docker run --rm -it --runtime=nvidia -v ${PWD}:/app -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix rin/torch:cuda-10.1-cudnn7-devel-ubuntu18.04

pip install numba pycuda
# pip install pycuda
python3 demo.py