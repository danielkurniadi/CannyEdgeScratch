// Header file | Canny Edge Detector demo
// canny.h
// 
// Created by: Daniel Kurniadi 
// Date: at 31 March 2020 (No April Fools)
// Copyright (c) 2020 Daniel Kurniadi. All rights reserved.


#ifndef _CANNY_
#define _CANNY_

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <vector>

#include "canny.h"

using namespace std;

class CannyEdgeDetector {
    private: 
        cv::Mat srcImage; // original image
        cv::Mat dstImage;  // grayscale image

        // define placeholder for processing pipeline output
        cv::Mat grayImage;
        cv::Mat gradientImage;     // gradient filtered image
        cv::Mat sobelImage;        // sobel filtered image
        cv::Mat angleMap;          // degree angle image/map
        cv::Mat nonMaxSuppImage;   // Non-maxima suppression image
        cv::Mat thresholdedImage;  // double thresholded and final

        // helper operations
        cv::Mat filterImage(cv::Mat &src, vector< vector<double> > &filter);

        // main operations
        void applyGaussianFilter(cv::Mat &src, cv::Mat &dst, cv::Size size, double sigma);
        void applySobelFilter(cv::Mat &src, cv::Mat &dst, cv::Mat &dstAngleMap);
        void applyNonMaxSuppression(cv::Mat &srcImage, cv::Mat &srcAngleMap, cv::Mat &dst);
        void applyHysteresisThreholding(cv::Mat &src, cv::Mat &dst, int , int);
        void hysteresisRecursion(cv::Mat &src, cv::Mat &dst, int x, int y, int lowThreshold);

        // make filters
        vector<vector<double> > createGaussianFilter(cv::Size size, double sigma);

    public:
        CannyEdgeDetector(std::string, cv::Size, double, int, int);  // constructor
};

#endif
