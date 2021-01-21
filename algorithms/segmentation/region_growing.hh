#pragma once

#include <Eigen/Core>
#include <iostream>

using namespace Eigen;
using namespace std;

namespace tpp {

    void region_growing(MatrixXf img, int neighborhood , int th)
    {
        int nrows = img.rows(), ncols = img.cols();
        MatrixXf visitedMatrix = MatrixXf::Zero(nrows, ncols);
        MatrixXf regionMatrix = MatrixXf::Zero(nrows, ncols);
        int currentRegionLabel = 1;

        VectorXi histogram[256];
        for (int row = 0; row < nrows; row++)
        {
            for (int col = 0; col < ncols; row++) {
                histogram[img(row,col)]++;
            }
        }



    }

    bool is_belonging_to_a_region(Vector2i point,MatrixXf img , std::vector<Vector2i> region_)
    {

    }

}
