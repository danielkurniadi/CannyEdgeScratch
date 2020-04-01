// Cpp file | Canny Edge Detector demo
// canny.cpp
// 
// Created by: Daniel Kurniadi 
// Date: at 1 April 2020 (No April Fools)
// Copyright (c) 2020 Daniel Kurniadi. All rights reserved.

#include <iostream>
#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>

#include "canny.h"

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"


CannyEdgeDetector::CannyEdgeDetector(std::string filepath, cv::Size gaussFilterSize,
                                     double gaussFilterSigma, int lowerThresh, int upperThresh)
{
    srcImage = cv::imread(filepath);

    if (!srcImage.data) // Check for invalid input
	{
		std::cout << "Image file not found in path: " <<  filepath << std::endl;
        return;
	}

    cv::Mat weakEdgeImage, strongEdgeImage;
    cv::Mat hysteresisImage; 

    cv::cvtColor(srcImage, grayImage, cv::COLOR_BGR2GRAY);

    // Step 1: apply Gaussian Filter (noise reduction)
    applyGaussianFilter(grayImage, gradientImage, gaussFilterSize, gaussFilterSigma); // gradientImage is of shape [H,W,1], pixel (0, 255)

    // Step 2: apply Sobel Filter
    applySobelFilter(gradientImage, sobelImage, angleMap); // sobelImage is of shape [H,W,1], pixel (0, 255)

    // Step 3: apply Non Maximum Suprression to edged image
    applyNonMaxSuppression(sobelImage, angleMap, nonMaxSuppImage); // nonMaxSuppImage is of shape [H,W,1], pixel (0, 255)

    // // Step 4: apply Double Thresholding
    // applyDoubleThresholding(nonMaxSuppImage, weakEdgeImage, strongEdgeImage, lowerThresh, upperThresh); // hysteresisImage is of shape [H,W,1], pixel (0, 255)

    // Step 5: apply Hysteresis Edge tracking
    applyHysteresisThreholding(nonMaxSuppImage, hysteresisImage, lowerThresh, upperThresh);

    dstImage = hysteresisImage;

};

/*
// filterImage()
// @param cv::Mat &srcImage - input matrix to be convolved upon
// @param <vector<vector<double>> filter
//
// @return <vector<vector<double>> gaussianFilter 
// - 2D gaussian filter with specified size and sigma
*/
cv::Mat CannyEdgeDetector::filterImage(cv::Mat &srcImage, vector< vector<double> > &filter)
{
    // Setup size and destination Image
    int size = (int)filter.size()/2;
	cv::Mat dstImage = cv::Mat(srcImage.rows - 2*size, srcImage.cols - 2*size, CV_8UC1);

	for (int i = size; i < srcImage.rows - size; i++)
	{
		for (int j = size; j < srcImage.cols - size; j++)
		{
			double sum = 0;
            
			for (int x = 0; x < filter.size(); x++)
				for (int y = 0; y < filter.size(); y++)
				{
                    sum += filter[x][y] * (double)(srcImage.at<uchar>(i + x - size, j + y - size));
				}

            dstImage.at<uchar>(i-size, j-size) = sum;
		}

	}
	return dstImage;
};

/*
// createGaussianFilter()
// @param cv:Size size - Size(width, height) of the kernel dimension
// @param double sigma - standard deviation (sigma) of the gaussian distribution
//
// @return <vector<vector<double>> gaussianFilter 
// - 2D gaussian filter with specified size and sigma
*/
vector<vector<double> > CannyEdgeDetector::createGaussianFilter(cv::Size size, double sigma)
{
    vector<vector<double>> gaussianFilter;

    int width = size.width;
    int height = size.height;

    // Create placeholder for gaussian filter
    for (int i = 0; i < height; i++)
	{
        vector<double> column;
        for (int j = 0; j < width; j++)
        {
            column.push_back(-1);
        }
		gaussianFilter.push_back(column);
	}

	float coordSum = 0;
	float constant = 2.0 * sigma * sigma;
	float sum = 0.0;

    // Create and calculate entries for 2D Gaussian Filter:
	for (int x = - height/2; x <= height/2; x++)
	{
		for (int y = -width/2; y <= width/2; y++)
		{
			coordSum = (x*x + y*y);
			gaussianFilter[x + height/2][y + width/2] = (exp(-(coordSum) / constant)) / (M_PI * constant);
			sum += gaussianFilter[x + height/2][y + width/2];
		}
	}

	// Normalize the Filter:
	for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
            gaussianFilter[i][j] /= sum;

	return gaussianFilter;
};

/*
// applyGaussianFilter()
// @param cv:Mat src - input matrix representing single channel image
// @param cv:Size size - Size(width, height) of the kernel dimension
// @param double sigma - standard deviation (sigma) of the gaussian distribution
//
// @returnparam cv::Mat dst
// - output matrix result of gaussian filter operation
*/
void CannyEdgeDetector::applyGaussianFilter(cv::Mat &src, cv::Mat &dst, cv::Size size, double sigma)
{
    vector<vector<double> >  gaussianFilter = createGaussianFilter(size, sigma);
    dst = filterImage(src, gaussianFilter);
    return;
};

/*
// applySobelFilter()
// @param cv:Mat src - input matrix representing single channel image
//
// @returnparam cv:Mat dst - output matrix result of sobel filter operation
// @returnparam cv:Mat dstAngleMap - output matrix representing the edge direction/angle
*/
void CannyEdgeDetector::applySobelFilter(cv::Mat &src, cv::Mat &dst, cv::Mat &dstAngleMap)
{
    // Sobel X Filter
    double x1[] = {-1.0, 0.0, 1.0};
    double x2[] = {-2.0, 0.0, 2.0};
    double x3[] = {-1.0, 0.0, 1.0};

    vector<vector<double>> xFilter(3);
    xFilter[0].assign(x1, x1+3);
    xFilter[1].assign(x2, x2+3);
    xFilter[2].assign(x3, x3+3);

    // Sobel Y Filter
    double y1[] = {1.0, 2.0, 1.0};
    double y2[] = {0, 0, 0};
    double y3[] = {-1.0, -2.0, -1.0};

    vector<vector<double>> yFilter(3);
    yFilter[0].assign(y1, y1+3);
    yFilter[1].assign(y2, y2+3);
    yFilter[2].assign(y3, y3+3);

    // Limit Size
    int size = (int)xFilter.size()/2;

	dst = cv::Mat(src.rows - 2*size, src.cols - 2*size, CV_8UC1);
    dstAngleMap = cv::Mat(src.rows - 2*size, src.cols - 2*size, CV_32FC1);

    // Sobel convolution
    for (int i = size; i < src.rows - size; i++)
	{   
		for (int j = size; j < src.cols - size; j++)
		{
			double sumx = 0;
            double sumy = 0;
            
            // dot operation with sobel filter
			for (int x = 0; x < xFilter.size(); x++)
				for (int y = 0; y < xFilter.size(); y++)
				{
                    sumx += xFilter[x][y] * (double)(src.at<uchar>(i + x - size, j + y - size)); // Sobel_X Filter Value
                    sumy += yFilter[x][y] * (double)(src.at<uchar>(i + x - size, j + y - size)); // Sobel_Y Filter Value
				}

            // calculate square sum of x and y filter output
            double sumxsq = sumx*sumx;
            double sumysq = sumy*sumy;
            double sq2 = sqrt(sumxsq + sumysq);

            if(sq2 > 255) //Unsigned char fix
                sq2 =255;
            dst.at<uchar>(i-size, j-size) = sq2;
 
            if(sumx == 0) // arctan fix, to avoid division with zero
                dstAngleMap.at<float>(i-size, j-size) = 90;
            else  // any possible direction 
                dstAngleMap.at<float>(i-size, j-size) = atan(sumy/sumx);
		}
	}

    return;
};

/*
// applyNonMaxSuppression()
// @param cv:Mat srcImage - input matrix representing single channel edged image obtained from sobel operation
// @param cv:Mat srcAngleMap - input matrix representing angle map obtained sobel operation
//
// @returnparam cv:Mat dst - output matrix result from non maximum suppresion
*/
void CannyEdgeDetector::applyNonMaxSuppression(cv::Mat &srcImage, cv::Mat &srcAngleMap, cv::Mat &dstImage)
{
    dstImage = cv::Mat(srcImage.rows-2, srcImage.cols-2, CV_8UC1);
    for (int i=1; i< srcImage.rows - 1; i++) {
        for (int j=1; j<srcImage.cols - 1; j++) {

            float Tangent = srcAngleMap.at<float>(i,j);
            dstImage.at<uchar>(i-1, j-1) = srcImage.at<uchar>(i,j);

            //Horizontal Edge
            if (((-22.5 < Tangent) && (Tangent <= 22.5)) || ((157.5 < Tangent) && (Tangent <= -157.5)))
            {
                if ((srcImage.at<uchar>(i,j) < srcImage.at<uchar>(i,j+1)) || (srcImage.at<uchar>(i,j) < srcImage.at<uchar>(i,j-1)))
                    dstImage.at<uchar>(i-1, j-1) = 0;
            }
            //Vertical Edge
            if (((-112.5 < Tangent) && (Tangent <= -67.5)) || ((67.5 < Tangent) && (Tangent <= 112.5)))
            {
                if ((srcImage.at<uchar>(i,j) < srcImage.at<uchar>(i+1,j)) || (srcImage.at<uchar>(i,j) < srcImage.at<uchar>(i-1,j)))
                    dstImage.at<uchar>(i-1, j-1) = 0;
            }
            
            //-45 Degree Edge
            if (((-67.5 < Tangent) && (Tangent <= -22.5)) || ((112.5 < Tangent) && (Tangent <= 157.5)))
            {
                if ((srcImage.at<uchar>(i,j) < srcImage.at<uchar>(i-1,j+1)) || (srcImage.at<uchar>(i,j) < srcImage.at<uchar>(i+1,j-1)))
                    dstImage.at<uchar>(i-1, j-1) = 0;
            }
            
            //45 Degree Edge
            if (((-157.5 < Tangent) && (Tangent <= -112.5)) || ((22.5 < Tangent) && (Tangent <= 67.5)))
            {
                if ((srcImage.at<uchar>(i,j) < srcImage.at<uchar>(i+1,j+1)) || (srcImage.at<uchar>(i,j) < srcImage.at<uchar>(i-1,j-1)))
                    dstImage.at<uchar>(i-1, j-1) = 0;
            }
        }
    }
    return;
}

void CannyEdgeDetector::applyHysteresisThreholding(cv::Mat &src, cv::Mat &dst, int lowerThresh, int upperThresh)
{
    dst = cv::Mat(src.rows, src.cols, CV_8UC1);
    
    for (int i = 0; i < dst.rows; i++) 
    {
        for (int j = 0; j < dst.cols; j++) 
        {
            if(src.at<uchar>(i,j) >= upperThresh)
            {
                dst.at<uchar>(i,j) = 255;
                hysteresisRecursion(src, dst, i, j, lowerThresh);
            }
            else if(src.at<uchar>(i,j) < lowerThresh)
                dst.at<uchar>(i,j) = 0;
        }
    }
};


void CannyEdgeDetector::hysteresisRecursion(cv::Mat &src, cv::Mat &dst, int x, int y, int lowerThresh)
{
	int value = 0;

	for (int i = x - 1; i <= x + 1; i++) {
		for (int j = y - 1; j <= y + 1; j++) {

			if ((i < src.rows) & (j < src.cols) & (i >= 0) & (j >= 0) & (i != x) & (j != y)) {
				value = src.at<uchar>(i, j);

				if (dst.at<uchar>(i,j) != 255) {
					if (value >= lowerThresh) {
						dst.at<uchar>(i,j) = 255;
						hysteresisRecursion(src, dst, i, j, lowerThresh);
                        return;
					}
					else {
						dst.at<uchar>(i, j) = 0;
					}
				}
			}
		}
	}
}