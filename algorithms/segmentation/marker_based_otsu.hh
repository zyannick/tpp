#pragma once

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
 
using namespace Eigen;

namespace tpp {

bool populate_marker_segmentation( std::vector<Vector2i> &marker, int row, int col, MatrixXf &mat , size_t i )
{
    int ncols = mat.cols();
    int nrows = mat.rows();
    int nb = 8;

    bool continue_recursion = true;

    mat(row, col) = 127;

    for(int c = col - 1; c <= col + 1; c ++ )
    {
        if(c < 0 || c >= ncols)
        {
            nb--;
            continue;
        }
        for(int r = row - 1 ; r <= row + 1; r++ )
        {
            if(r < 0 || r >= nrows)
            {
                nb--;
                continue;
            }
            nb--;
            if(mat(r, c) == 255)
            {
                Vector2i tmp_v;
                tmp_v.x() = c;
                tmp_v.y() = r;
                marker.push_back(tmp_v);
                populate_marker_segmentation(marker, r, c, mat, i);
            }
        }
    }


    return continue_recursion;
}

void merge_objects()
{

}


}
