#pragma once

#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include "cwise_functors.hpp"
#include <random>

#define _USE_MATH_DEFINES
#include <math.h>

 
using namespace Eigen;
using namespace std;



namespace tpp
{
    cv::Mat change_brightness_image(cv::Mat image, float alpha_, float beta_)
    {
        cv::Mat new_image = cv::Mat::zeros(image.rows, image.cols, image.type());
        for( int y = 0; y < image.rows; y++ ) {
            for( int x = 0; x < image.cols; x++ ) {
                for( int c = 0; c < image.channels(); c++ ) {
                    new_image.at<cv::Vec3b>(y,x)[c] =
                        cv::saturate_cast<uchar>( alpha_*image.at<cv::Vec3b>(y,x)[c] + beta_ );
                }
            }
        }
        return new_image;
    }

    cv::Mat read_16_converto_8(string path)
    {
        cv::Mat img_16;
        cv::Mat img;
        img_16 = cv::imread(path, CV_16U);
        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(img_16, &minVal, &maxVal, &minLoc, &maxLoc);
        img_16.convertTo(img, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
        //cv::resize(img, img, cv::Size(), 4, 4, INTER_CUBIC );
        return  img;
    }
    
    
}
