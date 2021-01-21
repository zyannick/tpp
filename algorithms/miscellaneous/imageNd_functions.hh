#pragma once

#include <vpp/vpp.hh>
#include <vpp/utils/opencv_bridge.hh>
#include <vpp/utils/opencv_utils.hh>
#include <math.h>

using namespace vpp;
using namespace std;
 
using namespace Eigen;


namespace tpp {

template<typename Type>
image2d<Type> magnitude(image2d<Type> X, image2d<Type> Y)
{
    image2d<Type> output = image2d<Type>(Y.nrows(), Y.ncols());
    pixel_wise(output, X, Y) | [&] (auto &o,auto x, auto y)
    {
        o = sqrt(x*x + y*y);
    };
    return output;
}

template<typename Type>
void nd_sin(image2d<Type> &img)
{
    pixel_wise(img) | [&] (auto &i)
    {
        i = sin(i);
    };
}


template<typename Type>
void nd_cos(image2d<Type> &img)
{
    pixel_wise(img) | [&] (auto &i)
    {
        i = cos(i);
    };
}

template<typename Type>
void nd_tan(image2d<Type> &img)
{
    pixel_wise(img) | [&] (auto &i)
    {
        i = tan(i);
    };
}

}
