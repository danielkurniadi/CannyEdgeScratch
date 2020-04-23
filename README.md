# Edge Detector (from scratch)

Canny and LoG edge detection source code. There are C++ code and Python code version to implement the canny edge detector and laplacian of gaussian. This is my student project.

Sample results from running the example

# Table of Contents
1. [Python Code Base](#Python-Code-Base)
    1. [Python Requirements](##Python-Requirements)
    2. [Python Demo: Usage and Setups](##Python-Demo:-Usage-and-Setups)
        1. [Canny Edge Detection Demo](###Canny-Edge-Detection-Demo)
        2. [Laplacian of Gaussian Demo](###Laplacian-of-Gaussian-Demo)
        3. [Jupyter edge detection notebook Demo](###Jupyter-edge-detection-notebook-Demo)
2. [C++ Code Base](#C++-Code-Base)
    1. [C++ Requirements](##C++-Requirements)
    2. [C++ Demo: Usage and Setups](##C++-Demo:-Usage-and-Setups)
        1. [Dockerfile](###1.-Dockerfile)
        2. [VSCode & Docker](###2.-VSCode-+-Docker)
        3. [ Build C++ Source and Run](###Build-C++-Source-and-Run)
3. [About Contributor](#About-Contributor)



# Python Code Base

The following sections discusses the setup, implementation and usage for python3 based edge detection
* Canny Edge Detection ([code](canny_edge_demo.py))
* Laplacian of Gaussian ([code](gaussian_laplace_demo.py))
* Jupyter notebook of both canny and LoG ([jupyter](edge_detection_notebook.ipynb))

This code and setups is only tested for `Debian/Linux` environment. Do modify and run on your own environment but 
there might be different configuration you need to find on your own for now. Currently I don't have MacOS or Windows OS to test.

## Python Requirements
* Python >= `3.6`
* Pip3 >= `19.2.3`

## Python Demo: Usage and Setups

First thing first, we need to install three packages as follows. Note that we still implement convolution and 
image operation from scratch. But we need to read, visualize and buffer for image arrays.

* Numpy >= `1.18.2`: Library for matrix and tensor data structure. Useful for matrix multiplication
* Pillow >= `7.1.1`: Library for loading image from raw or well known format.
* Matplotlib >= `3.1.2`: Library for visualization and plotting in the notebook.
* Jupyter >= `1.0.0` (optional): If you want to see jupyter python notebook. You can also use google colabs to view.

> **_NOTE:_** Bash command. If you see `$` in the terminal, don't forget to remove the `$`. 
Also modify or adjust variable in `<PATH/TO/PROJECT/FOLDER>` and remove the comment starting from `# comment ...`

Installation is performed as follows using bash terminal :


```bash
# go to this project folder, e.g. /user/Desktop/cannyEdgeScratch
$ cd <PATH/TO/PROJECT/FOLDER>  

$ python3 -V  # check if you're using Python3
$ pip3 --version  # check python package manager version
$ pip3 install -r requirements.txt
```

### Canny Edge Detection Demo

We can run the demo for canny-edge detection using our own image or image given in [data folder](data/)
Run the python script `canny_edge_demo.py` as follows:

```bash
# run script and give argument
python3 canny_edge_demo.py \
--in-file data/house.jpg  \  
--out-file result_canny_house.jpg \
--kernel-sigma 0.4 \
--lower-threshold 10 \
--upper-threshold 40
```

### Laplacian of Gaussian Demo

We can run the demo for laplacian of gaussian detection using our own image or image given in [data folder](data/)
Run the python script `gaussian_laplace_demo.py` as follows:

```bash
# run script and give argument
python3 gaussian_laplace_demo.py \
--in-file data/house.jpg \
--out-file result_log_house.jpg \
--kernel-size 9 \
--kernel-sigma 1.4
```

### Jupyter edge detection notebook Demo

If you have `jupyter` package in your Pip environment, you can run jupyter server to view `edge_detection_notebook.ipynb` 
as follows:
```bash
$ jupyter notebook --port=8899
```

You can also upload the `edge_detection_notebook.ipynb` to [google colabs](https://colab.research.google.com/) and view it there.


# C++ Code Base

The following sections documents the setup, implementation and usage for C++ based edge detection

Implemented in C++ Code are:
* Canny Edge Detection ([code](src/main.cpp))

## C++ Requirements

- OpenCV C++ (`3.4.3 >=`): We will only use `cv::Mat` to contain our flexible sized image array and pixel values. No other.
- Cmake and `build_essential`: Contain gcc compiler
- Docker Engine (optional): If you want to skip all the fuzz downloading OpenCV

## C++ Demo: Usage and Setups

There are several prefered way to reproduce the results and run the codebase. My favorite is using [Docker](https://docker.io) but you can choose for yourself.

### 1. Dockerfile

First, notice the `deploy/` directory has Dockerfile and bash script. It is for running docker at ease. We use `openCV3.4` docker image
thanks to @SatyaMalick. Here is the dockerhub repository for [spmallick/opencv-docker:opencv](https://hub.docker.com/r/spmallick/opencv-docker)

#### Build the docker image
We build the docker image from our Dockerfile.

```bash
docker build -f -t iqdf/cannyEdge:1.0 deploy/Dockerfile .
```

#### Run the docker container and access it through terminal

We would like to run the container, then persist it throughout the day so we can access the container environment
and play around with openCV 
```bash
# you can run using docker run command
docker run --device=/dev/video0:/dev/video0 \  
    -v /tmp/.X11-unix:/tmp/.X11-unix \  # in case you need cv::imshow()
    -e DISPLAY=$DISPLAY \
    -p 5000:5000 -p 8888:8888 \
    -it iqdf/cannyEdge:latest \
    /bin/bash
```

### 2. VSCode + Docker

Even easier. 

1. Just install Docker plugin in your VSCode and 
2. Hit `Open remote window` on your left bottom icon.
3. Then from dropdown, choose `Remote-connection:Reopen Folder in container`

You can edit this code base while playing or running the script. See the visual studio code documentation at [VSCode developing in Docker containers](https://code.visualstudio.com/docs/remote/containers)


## Build C++ Source and Run

Simply run below command and provide your own image data.
```bash
# build the source code
bash build_canny.sh

# run the compiled binary canny edge program
build/main \
    --infFile=<YOUR_IMAGE_PATH> --outDir=<YOUR_RESULT_DIR> \
    -gw=3 -gh=3 \
    --lowerThresh=20 \
    --upperThresh=80 \
    --sigma=1.4
```

# About Contributor

Please if you find the following implementation useful for your projects or educational purposes. Reference me in your project. Give this repository a star, fork, and donate to my account.

Copying and distribution of this file, with or without modification, are permitted in any medium without royalty provided the copyright notice and this notice are preserved. This file is offered as-is, without any warranty.

* Author    : Daniel Kurniadi [`@iqDF`](https://github.com/iqDF)
* Copyright : (C) 2020 Daniel Kurniadi
* Licence   : GPL v3+, see GPLv3 licence and [LICENCE.txt](LICENCE.txt)
