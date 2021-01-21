#pragma once

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <list>
#include "algorithms/interpolation/bilinear.hh"
using namespace std;
 

namespace tpp
{
void gravity_center(const cv::Mat img_in, cv::Mat& img_out, int wind)
{
    int ncols, nrows;
    ncols = img_in.cols;
    nrows = img_in.rows;
    int i = 0;
    for (int col = 0; col < ncols; col++)
        for (int row = 0; row < nrows; row++)
        {
            cv::Point2f coord = cv::Point2f(col, row);
            float sum_pixel = 0;
            float x_mid = 0;
            float y_mid = 0;
            for (auto x = coord.x - wind; x <= coord.x + wind; x++)
            {
                if (x > 0 && x < ncols)
                    for (auto y = coord.y - wind; y <= coord.y + wind; y++)
                    {
                        if (y > 0 && y < nrows)
                        {
                            sum_pixel += int(img_in.at<uchar>(y, x));
                            i++;
                        }
                    }
            }
            img_out.at<double>(coord) = sum_pixel / (float)i;
            i = 0;
        }
}

void bilinear_interpolation(const cv::Mat img_in, cv::Mat& img_out)
{
    int ncols, nrows;
    ncols = img_in.cols;
    nrows = img_in.rows;
    for (int col = 1; col < ncols - 1; col++)
    {
        for (int row = 1; row < nrows - 1; row++)
        {
            cv::Point2f coord = cv::Point2f(col, row);
            //cout << "here " << col << "  " << row << endl;
            img_out.at<double>(coord) = (bilinear<double>(coord, img_in)).gray;
        }
    }
}

void get_local_maxima(int wind, const cv::Mat img, std::list<cv::Point2f> list_temp, std::vector<cv::Point2f> &imagePoints)
{
    int ncols, nrows;
    ncols = img.cols;
    nrows = img.rows;
    int i = 0;
    for (auto &l : list_temp)
    {
        cv::Point2f coord = l;
        float max = 0;
        cv::Point2f pt_max = cv::Point2f(0, 0);
        for (auto x = coord.x - wind; x <= coord.x + wind; x++)
        {
            if (x > 0 && x < ncols)
                for (auto y = coord.y - wind; y <= coord.y + wind; y++)
                {
                    if (y > 0 && y < nrows)
                    {
                        if (max < img.at<double>(cv::Point2f(x, y)))
                        {
                            pt_max = cv::Point2f(x, y);
                            max = img.at<double>(cv::Point2f(x, y));
                        }
                    }
                }
        }
        imagePoints[i] = pt_max;
        i++;
    }
}

void get_local_maxima_gc(int wind, cv::Mat img, std::list<cv::Point2f> list_temp, std::vector<cv::Point2f> &imagePoints)
{
    int i = 0;
    int ncols, nrows;
    ncols = img.cols;
    nrows = img.rows;

    for (auto &l : list_temp)
    {
        cv::Point2f coord = l;
        float sum_pixel = 0;
        float x_mid = 0;
        float y_mid = 0;
        for (auto x = coord.x - wind; x <= coord.x + wind; x++)
        {
            if (x > 0 && x < ncols)
                for (auto y = coord.y - wind; y <= coord.y + wind; y++)
                {
                    //cout << "coord " << x << "   " << y << endl;
                    if (y > 0 && y < nrows)
                    {
                        x_mid += x * int(img.at<uchar>(y, x));
                        y_mid += y * int(img.at<uchar>(y, x));
                        sum_pixel += int(img.at<uchar>(y, x));
                        //cout << "x_mid " << x << " y_mid " << y << " sum_pixel " << int(img_8bit.at<uchar>(y, x)) << endl;
                    }
                }
        }
        //cout << "sum_pixel  " << sum_pixel << endl;
        float x_ = x_mid / sum_pixel;
        float y_ = y_mid / sum_pixel;
        imagePoints[i] = cv::Point2f(x_, y_);
        //cout << "klklj " << i << endl;
        i++;
    }
}


void get_local_maxima_gc(int wind, cv::Mat img, cv::Point2f coord, cv::Point2f &coord_gc)
{
    int ncols, nrows;
    ncols = img.cols;
    nrows = img.rows;
    float sum_pixel = 0;
    float x_mid = 0;
    float y_mid = 0;
    for (auto x = coord.x - wind; x <= coord.x + wind; x++)
    {
        if (x > 0 && x < ncols)
            for (auto y = coord.y - wind; y <= coord.y + wind; y++)
            {
                if (y > 0 && y < nrows)
                {
                    x_mid += x * int(img.at<uchar>(y, x));
                    y_mid += y * int(img.at<uchar>(y, x));
                    sum_pixel += int(img.at<uchar>(y, x));
                }
            }
    }
    float x_ = x_mid / sum_pixel;
    float y_ = y_mid / sum_pixel;
    coord_gc = cv::Point2f(x_, y_);
}

cv::Mat getGaussianKernel(int rows, int cols, double sigmax, double sigmay)
{
    cv::Mat kernel = cv::Mat::zeros(rows, cols, CV_32FC1);
    double eps = 0.00001;
    float meanj = (kernel.rows - 1) / 2,
            meani = (kernel.cols - 1) / 2,
            sum = 0,
            temp = 0;

    int sigma = 2 * sigmay*sigmax;
    for (unsigned j = 0; j < kernel.rows; j++)
        for (unsigned i = 0; i < kernel.cols; i++)
        {
            temp = exp(-((j - meanj)*(j - meanj) + (i - meani)*(i - meani)) / (sigma));
            if (temp > eps)
                kernel.at<float>(j, i) = temp;

            sum += kernel.at<float>(j, i);
        }

    if (sum != 0)
        return kernel /= sum;
    else return cv::Mat();
}

}
