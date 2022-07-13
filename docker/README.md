#### Usage

```bash
cd tsdf-fusion-python/docker/
```

##### build docker image

build image(this might take some time)
```bash
./docker.sh build
```

##### run container using docker image(-v option is to mount directory)
```bash
./docker.sh run
```

If you don't have a GPU, you can run
```bash
./docker.sh run_cpu
```
<br><br>





docker run -it --rm -v ${PWD}:/app rin/torch:cuda-10.1-cudnn7-devel-ubuntu18.04
pip install --user numba pycuda