#pragma once

#include <iostream>
#include <Eigen\core>

using namespace Eigen;
using namespace std;

namespace tpp
{
    void shapiro_wilk_test(MatrixXf H, VectorXf p_value, VectorXf W, VectorXf x, float alpha )
    {
        assert(x.rows() > 3 && "We need at least 3 valid observations");
        assert(x.rows() < 5000 && "Shapiro-Wilk test does not work well with more than 5000 observations");
        std::sort(x.data(), x.data()+x.size());
        int n = x.rows();
        VectorXf m_tilde = VectorXf::Zero(n);
        VectorXf temp(n);
        VectorXf three_height = VectorXf::Constant(n,3.0/8.0);
        for(int i = 0 ; i < n ; i++)
            temp(i) = i+1;
        temp = temp - three_height;
        temp = temp / (n + 1.0/4.0);


    }


}
