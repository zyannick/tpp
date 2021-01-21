#pragma once

#define NOMINMAX

#include <Eigen/Core>
#include <limits>
#include  <iostream>

using namespace Eigen;


namespace tpp {

void thinningIteration(MatrixXf& im, int iter)
{

    int nrows = im.rows(), ncols = im.cols();

    MatrixXf marker = MatrixXf::Zero( nrows, ncols );

    for (int i = 1; i < nrows-1; i++)
    {
        for (int j = 1; j < ncols-1; j++)
        {
            float p2 = im(i-1, j);
            float p3 = im(i-1, j+1);
            float p4 = im(i, j+1);
            float p5 = im(i+1, j+1);
            float p6 = im(i+1, j);
            float p7 = im(i+1, j-1);
            float p8 = im(i, j-1);
            float p9 = im(i-1, j-1);

            int A  = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
                     (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
                     (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                     (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
            int B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
            int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
            int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);

            if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
                marker(i,j) = 1;
        }
    }

    im = marker;
}

void thinning(MatrixXf src, MatrixXf& dst)
{
    dst = src;

    MatrixXf prev = MatrixXf::Zero(src.rows(), src.cols());
    MatrixXf diff;

    do {
        thinningIteration(dst, 0);
        thinningIteration(dst, 1);
        dst = (prev - diff).cwiseAbs();
        dst = prev;
    }
    while (diff.maxCoeff() > 0);

}

}


