//
//  main.cpp
//  Canny Edge Detector
//
//  Created by Hasan Akgün on 21/03/14.
//  Copyright (c) 2014 Hasan Akgün. All rights reserved.
//

#include <iostream>
#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "canny.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    // ------------------------
    // ARGUMENT PARSING
    // ------------------------

    // IO operation
    const cv::String keys =
        "{ i inFile        |    |  filepath of input image }"
        "{ o outDir        |    |  filepath of output image }"
        "{ gw gaussWidth   | 3  |  width of gaussian kernel in px }"
        "{ gh gaussHeight  | 3  |  height of gaussian kernel in px }"
        "{ s sigma         | 1.0 | specify the sigma of gaussian kernel }"
        "{ l lowerThresh   | 80  | specify lower threshold for hysteresis }"
        "{ u upperThresh   | 120 | specify upper threshold for hysteresis }"
    ;
    
    cv::CommandLineParser cmd(argc, argv, keys);

    std::string inFile        = cmd.get<std::string>("inFile");
    std::string outFile       = cmd.get<std::string>("outDir");
    int gaussWidth            = cmd.get<int>("gaussWidth");
    int gaussHeight           = cmd.get<int>("gaussHeight");
    double sigma              = cmd.get<double>("sigma");
    int lowerThresh           = cmd.get<int>("lowerThresh");
    int upperThresh           = cmd.get<int>("upperThresh");

    if (!cmd.check()){
        cmd.printErrors();
        return EXIT_FAILURE;
    }

    // ------------------------
    // MAIN 
    // ------------------------
    cv::Size size(gaussWidth, gaussHeight);
    CannyEdgeDetector canny(inFile, size, sigma, lowerThresh, upperThresh);

    return 0;
}
