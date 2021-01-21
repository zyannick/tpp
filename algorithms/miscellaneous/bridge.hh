#pragma once

#include <Eigen/Core>
#include <opencv2/opencv.hpp>


using namespace Eigen;
using namespace std;

namespace tpp {

template<typename T>
inline
Mat convert_eigen_to_mat(Matrix<T,Dynamic ,Dynamic> img,int chan)
{
    Mat mat;
    if(chan==1)
    {
        mat = Mat(img.rows(), img.cols(), CV_8U);
    }
    else if(chan==3)
    {
        mat = Mat(img.rows(), img.cols(), CV_8UC3);
    }

    for (int row = 0; row < img.rows(); row++)
    {
        for (int col = 0; col < img.cols(); col++)
        {
            if(chan == 1)
            mat.at<uchar>(row, col) = uchar(img(row, col));
            else if(chan == 3)
            mat.at<Vec3b>(row, col) = Vec3b(img(row, col),img(row, col),img(row, col));
        }
    }
    return mat;
}


template<typename T>
inline
Matrix<T,Dynamic ,Dynamic> convert_mat_to_eigen(int chan, Mat mat)
{

}

}
