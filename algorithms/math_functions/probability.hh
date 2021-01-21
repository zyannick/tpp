#pragma once


#include <Eigen/Core>
#include <iostream>


using namespace Eigen;
using namespace std;

namespace tpp {


    enum class TYPE_VALUE  { VARIANCE ,  STANDARD_DEVIATION , MEAN };


    float variance_vector(VectorXf vect_)
    {
        float moy = vect_.mean();
        float var = 0;
        for(int i = 0; i < vect_.rows() ; i++)
        {
            var += pow(vect_[i] - moy,2);
        }
        var /= vect_.rows();
        return var;
    }


}
