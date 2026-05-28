This docker image/container is made to install gtsam, modify the python bindings, to allow you to install it elsewhere without needing to install any of the other gtsam dependencies on your host computer.

# How To Use

## Docker 

1. Edit the compose file to volume in folders that you need. (Dockerfile and compose file are copied at end of this file)

2. Build the docker image.
```bash
docker compose -d -f compose.yaml --build
```

3. Enter the container 
```bash
docker exec -it gtsam_test bash
```
or you can connect vscode to it by attaching it to a container (google it if you don't know how to do that)

## Edit the python bindings and rebuild

1. Use nano or vscode or a volume or something to edit which ever python bindings you need to edit/add. Usually the files you'll need to edit for this end in .i

2. Rebuild gtsam

```bash
cd /home/gtsam/build
make -j
```

## Build the python wheel and repair it to make it portable

1. Build the python wheel
```bash
cd ./python
pip install wheel setuptools auditwheel
python3 setup.py bdist_wheel
cd ./dist
auditwheel repair gtsam-*.whl
```

## Move wheel outside of docker to use in a python environment

1. Move to location where you want to put the wheel on your host computer and copy the wheel out of docker
```bash
docker cp gtsam_test:/home/gtsam/build/python/dist/wheelhouse/gtsam*.whl .
```

2. Activate your python environment using `conda`, `mamba`, or `venv`
```bash
mamba create -n gtsam python=3.12 pip numpy scipy matplotlib
mamba activate gtsam
```

3. Install the wheel
```bash 
pip install gtsam*.whl
```

4. Test to make sure that it works
```bash
python -c "import gtsam; print(gtsam.__file__)"
```

5. Check to make sure that your python binding worked as well. Append any subclass needed to gtsam to make sure it works. (e.g. if you added a python binding in ISAM2, you would run `dir(gtsam.IAM2)` and check to make sure the appropriate function is inside of there)
```bash
python -c "import gtsam; dir(gtsam)"
```



# Docker file for reference

Below is the docker file used for above followed by the docker compose file.

`dockerfile`:
```dockerfile
FROM ubuntu:24.04

# Set up python installation
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12-full \
    python3.12-dev \
    python3-pip
# Set this environment variable to allow pip installation to 'system' python. Virtual environments might be a more elegant solution, but that's for another time. This is my 'quick' fix.
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# GTSAM prereqs
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libboost-all-dev \
    libtbb-dev \
    make \
    cmake \
    wget \
    patchelf

# Download GTSAM
RUN cd /home && \
    wget -O gtsam.tar.gz https://github.com/borglab/gtsam/archive/82fcedf.tar.gz && \
    mkdir gtsam && \
    tar -zxf gtsam.tar.gz -C gtsam --strip-components 1 && \
    rm gtsam.tar.gz

# Install python package prereqs and any desired libraries
RUN pip install -r /home/gtsam/python/dev_requirements.txt && \
    pip install matplotlib ipython scipy

# Install/Setup GTSAM for both C++ and Python
#-DGTSAM_PYTHON_VERSION=3.12
RUN cd /home/gtsam && \
    mkdir build && \
    cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=/usr -DGTSAM_BUILD_PYTHON=1 && \
    make install -j4 && \
    # make python-install
    cd python && \
    pip install . && \
    ldconfig

# Copy local files to container
WORKDIR /root

```


`compose.yaml`:
```yaml
services:
  gtsam_test:
    build:
      context: .
      dockerfile: dockerfile
    image: gtsam_test
    container_name: gtsam_test
    volumes:
      - ${HOME}:/home/user # Change this as needed
    network_mode: host
    stdin_open: true
    tty: true
    working_dir: /home/gtsam
    command: ["bash"]
```