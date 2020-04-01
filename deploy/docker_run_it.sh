#! /bin/bash

# docker pull image from spmallick's dockerhub
docker pull spmallick/opencv-docker:opencv

# docker run image
docker run --device=/dev/video0:/dev/video0 \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-e DISPLAY=$DISPLAY \
-p 5000:5000 -p 8888:8888 \
-it spmallick/opencv-docker:opencv \
/bin/bash
