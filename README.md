# Canny Edge Detector (from scratch)

Some C++ code to implement canny edge detector. This is my student project. Some parts of code may be influenced by various examples found on internet.


## Requirements

- OpenCV C++ (`3.4.3 >=`)
- Cmake

## Usage and Setups

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

## 2. VSCode + Docker

Even easier. 

1. Just install Docker plugin in your VSCode and 
2. Hit `Open remote window` on your left bottom icon.
3. Then from dropdown, choose `Remote-connection:Reopen Folder in container`

You can edit this code base while playing or running the script.


### Build Source and Run

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