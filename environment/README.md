# How to Add Python Bindings to gtsam and Repackage It

This docker image/container is made to install gtsam, modify the python bindings, to allow you to install it elsewhere without needing to install any of the other gtsam dependencies on your host computer.

  
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

Example: Add isam2.marginalizeLeaves
	To do this, I added the line 
	```
	void marginalizeLeaves(const gtsam::FastList<gtsam::Key>& leafKeys);
	```
	 to the `gtsam->nonlinear->nonlinear.i` file in the `class ISAM2 {}` section. 
	 If you need help figuring out how to do this, I found deepwiki.com to be very useful. Put in the gtsam github link and tell it what function you want to bind and it should be able to direct you with which file to edit and what to add where.

2. Rebuild gtsam

  

```bash
mkdir /home/gtsam/build
cd /home/gtsam/build

cmake .. -DGTSAM_BUILD_PYTHON=ON -DPYTHON_EXECUTABLE=$(which python3)
make python-install -j 4

```

  

## Build the python wheel and repair it to make it portable

  

1. Build the python wheel

```bash

cd ./python

pip install wheel setuptools auditwheel

# python3 setup.py bdist_wheel # Old way. Doesn't have .pyi files
# cd ./dist

pip wheel . -w wheelhouse

auditwheel repair wheelhouse/gtsam-*.whl

```

2. If you want vscode to auto complete functions or give hints for function inputs, make sure the wheel has .pyi files in it
```bash
apt update
apt install unzip

unzip -l wheelhouse/gtsam-4.3a0-py3-none-manylinux_2_39_x86_64.whl | grep pyi
```
  > Replace the name of the gtsam wheel as needed.

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

pip install gtsam*.whl --force-reinstall
# pip install gtsam-stubs not needed if you have .pyi files in your wheel

```
Pylance in vscode won't know what gtsam functions are valid and such and will say that they are all errors even though they will run just fine. The gtsam-stubs will tell Pylance all of the default gtsam functions, however, your custom bindings will still show as errors even though they should run just fine if done properly.
  

4. Test to make sure that it works

```bash

python -c "import gtsam; print(gtsam.__file__)"

```

  

5. Check to make sure that your python binding worked as well. Append any subclass needed to gtsam to make sure it works. (e.g. if you added a python binding in ISAM2, you would run `dir(gtsam.IAM2)` and check to make sure the appropriate function is inside of there)

```bash

python -c "import gtsam; dir(gtsam)"

```
