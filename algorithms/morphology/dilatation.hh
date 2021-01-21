#pragma once


#define NOMINMAX

#include <Eigen/Core>
#include <limits>
#include  <iostream>

using namespace Eigen;



namespace tpp {

template<typename Derived>
void dilate_image(MatrixBase<Derived> src, MatrixBase<Derived> &dst, MatrixBase<Derived> kernel_ )
{
    int nrows,ncols, k_nrows, k_ncols;
    nrows = src.rows();
    ncols = src.cols();
    k_nrows = kernel_.rows();
    k_ncols = kernel_.cols();
    int m_r = int(floor(k_nrows/2)) , m_c = int(floor(k_ncols));
    for(int row = 0 ; row < nrows; row++)
    {
        for(int col = 0 ; col < ncols ; col++)
        {
            for(int y = row - m_r,y_k = 0 ; y <= row + m_r ; y++, y_k++)
            {
                for(int x = col - m_c, x_k =0; x <= col + m_c ; x++, x_k++)
                {
                    if(y >=0 && y < nrows && x >= 0 && x < ncols)
                    {
                        dst(y,x) = max(src(y,x), kernel_(y,x));
                    }
                }
            }
        }
    }

}

}

