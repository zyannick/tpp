
#pragma once

#include <Eigen/Dense>
#include <Eigen/Core>
#include <list>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "core.hpp"
#include "phase_correlation.hh"

 
using namespace Eigen;
using namespace tpp;


namespace tpp {

void fill_border_zero(MatrixXf src,MatrixXf &dst, int pad)
{
    int ncols = src.cols();
    int nrows = src.rows();
    int new_ncols = ncols + 2*pad;
    int new_nrows = nrows + 2*pad;
    dst = MatrixXf::Zero(new_nrows,new_ncols);
    dst.block(pad,pad,nrows,ncols) = src;
}

inline
bool is_same_match(stereo_match left_right_match,stereo_match right_left_match, Vector2d error_matching ,bool is_stereo = true )
{
    Vector2d left_1 = left_right_match.first_point;
    Vector2d right_1 = left_right_match.second_point;

    Vector2d left_2 = right_left_match.first_point;
    Vector2d right_2 = right_left_match.second_point;

    float err_y = error_matching.y();
    float err_x = error_matching.x();

    if(fabs(left_1.y() - left_2.y()) < err_y && fabs(left_1.x() - left_2.x()) < err_x )
    {
        if(fabs(right_1.y() - right_2.y()) < err_y && fabs(right_1.x() - right_2.x()) < err_x )
        {
            return true;
        }
    }
    return false;
}


}
