#pragma once

#include <Eigen/Core>
#include <iostream>

using namespace Eigen;
using namespace std;


namespace tpp
{


int otsu_segmentation(MatrixXf img, int gray_level = 255)
{
    //cout << "otsu_segmentation start" << endl;

    VectorXi histogram = VectorXi::Zero(gray_level+1);
    VectorXf probability = VectorXf::Zero(gray_level+1);
    VectorXf myu = VectorXf::Zero(gray_level+1);
    VectorXf omega = VectorXf::Zero(gray_level+1);
    VectorXf sigma = VectorXf::Zero(gray_level+1);
    double max_sigma; // inter-class variance
    // int i, x, y; //Loop variable
    int threshold; //threshold for binarization
    int nrows = img.rows(), ncols = img.cols();

    //cout << "histogram" << endl;
    for (int row = 0; row < nrows; row++)
    {
        for(int col = 0 ; col < ncols ; col++  )
        {
            //assert(img(row,col) < gray_level);
            //cout << "hits " << img(row,col) << endl;
            histogram[int(img(row,col))]++;

        }
    }

    //cout << "calculation of probability density" << endl;
    //calculation of probability density
    for (int i = 0; i < gray_level; i ++ ) {
        probability[i] = (float)histogram[i] / (nrows * ncols);
    }



    //omega & myu generation
    //cout << "omega & myu generation" << endl;
    omega[0] = probability[0];
    myu[0] = 0.0;       /* 0.0 times prob[0] equals zero */
    for (int i = 1; i < gray_level; i++) {
        omega[i] = omega[i-1] + probability[i];
        myu[i] = myu[i-1] + i*probability[i];
    }

    //sigma maximization
    //sigma stands for inter-class variance
    //and determines optimal threshold value
    //cout << "sigma maximization" << endl;
    threshold = 0;
    max_sigma = 0.0;
    for (int i = 0; i < gray_level-1; i++) {
        if (omega[i] != 0.0 && omega[i] != 1.0)
            sigma[i] = pow(myu[gray_level-1]*omega[i] - myu[i], 2) /
                    (omega[i]*(1.0 - omega[i]));
        else
            sigma[i] = 0.0;
        if (sigma[i] > max_sigma) {
            max_sigma = sigma[i];
            threshold = i;
        }
    }

    //cout << "otsu_segmentation end" << endl;

    return threshold;

}


}
