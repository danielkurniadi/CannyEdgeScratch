FROM spmallick/opencv-docker:opencv
LABEL maintainer="Daniel Kurniadi dkurniad001@e.ntu.edu.sg"

# copy necessary source code and scripts to docker environment
COPY . /canny_edge_detector
COPY . CMakeLists.txt
COPY . build_canny_edge.sh
WORKDIR /canny_edge_detector
