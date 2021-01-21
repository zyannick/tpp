#pragma once

#define NOMINMAX

#include <Eigen/Core>
#include <limits>
#include  <iostream>

using namespace Eigen;


namespace tpp {

void hysteresis_thresholding(MatrixXf img , float T1, float T2)
{
    float min_,max_;
    min_ = std::min(T1,T2);
    max_ = std::max(T1,T2);


}

}
