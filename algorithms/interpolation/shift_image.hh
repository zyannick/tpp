#pragma once

/*
 *  Software License Agreement (BSD License)
 *
 *  Copyright (c) 2012, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  File:    shift.hpp
 *  Author:  Hilton Bristow
 *  Created: Aug 23, 2012
 */

#include <limits>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "algorithms/miscellaneous.hh"

/*! @brief shift the values in a matrix by an (x,y) offset
 *
 * Given a matrix and an integer (x,y) offset, the matrix will be shifted
 * such that:
 *
 *  src(a,b) ---> dst(a+y,b+x)
 *
 * In the case of a non-integer offset, (e.g. cv::Point2f(-2.1, 3.7)),
 * the shift will be calculated with subpixel precision using bilinear
 * interpolation.
 *
 * All valid OpenCV datatypes are supported. If the source datatype is
 * fixed-point, and a non-integer offset is supplied, the output will
 * be a floating point matrix to preserve subpixel accuracy.
 *
 * All border types are supported. If no border type is supplied, the
 * function defaults to BORDER_CONSTANT with 0-padding.
 *
 * The function supports in-place operation.
 *
 * Some common examples are provided following:
 * \code
 *  // read an image from file
 *  Mat mat = imread(filename);
 *  Mat dst;
 *
 *  // Perform Matlab-esque 'circshift' in-place
 *  shift(mat, mat, Point(5, 5), BORDER_WRAP);
 *
 *  // Perform shift with subpixel accuracy, padding the missing pixels with 1s
 *  // NOTE: if mat is of type CV_8U, then it will be converted to type CV_32F
 *  shift(mat, mat, Point2f(-13.7, 3.28), BORDER_CONSTANT, 1);
 *
 *  // Perform subpixel shift, preserving the boundary values
 *  shift(mat, dst, Point2f(0.093, 0.125), BORDER_REPLICATE);
 *
 *  // Perform a vanilla shift, integer offset, very fast
 *  shift(mat, dst, Point(2, 2));
 * \endcode
 *
 * @param src the source matrix
 * @param dst the destination matrix, can be the same as source
 * @param delta the amount to shift the matrix in (x,y) coordinates. Can be
 * integer of floating point precision
 * @param fill the method used to fill the null entries, defaults to BORDER_CONSTANT
 * @param value the value of the null entries if the fill type is BORDER_CONSTANT
 */

using namespace std;
 

namespace tpp
{


void shift(const cv::Mat& src, cv::Mat& dst, cv::Point2f delta, int fill=cv::BORDER_CONSTANT, cv::Scalar value=cv::Scalar(0,0,0,0)) {

    // error checking
    assert(fabs(delta.x) < src.cols && fabs(delta.y) < src.rows);

    // split the shift into integer and subpixel components
    cv::Point2i deltai(ceil(delta.x), ceil(delta.y));
    cv::Point2f deltasub(fabs(delta.x - deltai.x), fabs(delta.y - deltai.y));

    // INTEGER SHIFT
    // first create a border around the parts of the Mat that will be exposed
    int t = 0, b = 0, l = 0, r = 0;
    if (deltai.x > 0) l =  deltai.x;
    if (deltai.x < 0) r = -deltai.x;
    if (deltai.y > 0) t =  deltai.y;
    if (deltai.y < 0) b = -deltai.y;
    cv::Mat padded;
    cv::copyMakeBorder(src, padded, t, b, l, r, fill, value);

    // SUBPIXEL SHIFT
    float eps = std::numeric_limits<float>::epsilon();
    if (deltasub.x > eps || deltasub.y > eps) {
        switch (src.depth()) {
            case CV_32F:
            {
                cv::Matx<float, 1, 2> dx(1-deltasub.x, deltasub.x);
                cv::Matx<float, 2, 1> dy(1-deltasub.y, deltasub.y);
                sepFilter2D(padded, padded, -1, dx, dy, cv::Point(0,0), 0, cv::BORDER_CONSTANT);
                break;
            }
            case CV_64F:
            {
                cv::Matx<double, 1, 2> dx(1-deltasub.x, deltasub.x);
                cv::Matx<double, 2, 1> dy(1-deltasub.y, deltasub.y);
                sepFilter2D(padded, padded, -1, dx, dy, cv::Point(0,0), 0, cv::BORDER_CONSTANT);
                break;
            }
            default:
            {
                cv::Matx<float, 1, 2> dx(1-deltasub.x, deltasub.x);
                cv::Matx<float, 2, 1> dy(1-deltasub.y, deltasub.y);
                padded.convertTo(padded, CV_32F);
                sepFilter2D(padded, padded, CV_32F, dx, dy, cv::Point(0,0), 0, cv::BORDER_CONSTANT);
                break;
            }
        }
    }

    // construct the region of interest around the new matrix
    cv::Rect roi = cv::Rect(max(-deltai.x,0),max(-deltai.y,0),0,0) + src.size();
    dst = padded(roi);/**/
}


void shift_list_images(bool save = false)
{
    std::vector<string> left_images, right_images;
    string inputFilename;
    inputFilename = std::string("./json/bats.json");
    remplirListes_absolute(inputFilename, left_images, right_images,std::string("./Images/Bats/Original/"));
    int nimages = left_images.size();

    for (int ster = 0; ster < nimages; ster++)
    {

        cout << "image numero " << ster << endl;

        string left_image =left_images[ster];
        cv::Mat left_view_16 = cv::imread(left_image, CV_16U);
        cv::Mat left_view;

        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        minMaxLoc(left_view_16, &minVal, &maxVal, &minLoc, &maxLoc);
        left_view_16.convertTo(left_view, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
        cv::Mat dst;
        float pad_x = 5.523;
        float pad_y = 0;
        shift(left_view, dst, cv::Point2f(pad_x,pad_y), cv::BORDER_REPLICATE);

        int nrows = left_view.rows;
        int ncols = left_view.cols;

        /*Mat res_src = Mat::zeros(nrows,ncols-int(pad_x),CV_8U);
        Mat res_dst = Mat::zeros(nrows,ncols-int(pad_x),CV_8U);

        for(int row = 0 ; row < nrows; row++)
        {
            for(int col = int(pad_x) , c = 0 ; col < ncols ; col++, c++)
            {
                res_dst.at<uchar>(row,c) = dst.at<uchar>(row,col);
            }
            for(int col = 0 , c = 0 ; col < ncols-int(pad_x) ; col++, c++)
            {
                res_src.at<uchar>(row,c) = dst.at<uchar>(row,col);
            }
        }*/

        if(save)
        {
            string file_src;
            file_src = std::string("./Images/Bats/_5.523/Camera1/original_").append(std::to_string(ster)).append(".bmp");
            string file_dst;
            file_dst = std::string("./Images/Bats/_5.523/Camera2/shift_").append(std::to_string(ster)).append(".bmp");
            if(!imwrite(file_src,left_view))
            {
              cout << "impossible" << endl;
            }
            imwrite(file_dst,dst);
        }


        //return;
    }
}

}
