#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>

#define _USE_MATH_DEFINES
#include <math.h>


using namespace std;
using namespace Eigen;

namespace tpp {

template<typename T>
/**
 * @brief gaussian_kernel_iso
 * @param wx
 * @param wy
 * @param sigma
 * @return
 */
Matrix<T, Dynamic, Dynamic> gaussian_kernel_iso(int wx, int wy, T sigma)
{
    assert(wx == wy);
    int n = wx;
    Matrix<T, Dynamic, Dynamic> Gaussian_kernel_ = Matrix<T, Dynamic, Dynamic>::Zero(n, n);
    T half_n = (n - 1) / 2.0;
    T sigma_squared = sigma * sigma;
    for (int col = 0;  col < n;  ++col)
    {
        T x = col - half_n;
        for (int row = 0;  row < n;  ++row)
        {
            T y = row - half_n;
            Gaussian_kernel_(row, col) = (1 / (2* M_PI * sigma_squared)) * exp(- (x*x + y*y)/(2* sigma_squared)) ;
        }
    }
    return Gaussian_kernel_;
}

}
