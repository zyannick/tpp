#pragma once

#ifdef vpp
#include <vpp/vpp.hh>
#include <vpp/utils/opencv_bridge.hh>
#include <vpp/utils/opencv_utils.hh>
using namespace vpp;
#endif


using namespace std;
 
using namespace Eigen;


namespace tpp {

#ifdef vpp

template< typename T>
image2d<T> operator +(T value, image2d<T> A)
{
    image2d<T> B(A.nrows(), A.ncols());
    pixel_wise(A, B) | [&] (auto a, auto &b)
    {
        b = a + value;
    };
    return B;
}

template< typename T>
image2d<T> operator *(T value, image2d<T> A)
{
    image2d<T> B(A.nrows(), A.ncols());
    pixel_wise(A, B) | [&] (auto a, auto &b)
    {
        b = a * value;
    };
    return B;
}

template< typename T>
image2d<T> operator /(T value, image2d<T> A)
{
    image2d<T> B(A.nrows(), A.ncols());
    pixel_wise(A, B) | [&] (auto a, auto &b)
    {
        b = value / a;
    };
    return B;
}

template< typename T>
image2d<T> operator /(image2d<T> A, T value)
{
    image2d<T> B(A.nrows(), A.ncols());
    pixel_wise(A, B) | [&] (auto a, auto &b)
    {
        b = a / value;
    };
    return B;
}

template< typename T>
image2d<std::complex<T>> operator /(image2d<std::complex<T>> A, T value)
{
    image2d<std::complex<T>> B(A.nrows(), A.ncols());
    pixel_wise(A, B) | [&] (auto a, auto &b)
    {
        b = a / value;
    };
    return B;
}

template< typename T>
image2d<T> operator +(image2d<T> A, image2d<T> B)
{
    image2d<T> C(A.nrows(), A.ncols());
    pixel_wise(A, B, C) | [&] (auto a, auto b, auto &c)
    {
        c = a + b;
    };
    return C;
}

template< typename T>
image2d<T> operator *(image2d<T> A, image2d<T> B)
{
    image2d<T> C(A.nrows(), A.ncols());
    pixel_wise(A, B, C) | [&] (auto a, auto b, auto &c)
    {
        c = a * b;
    };
    return C;
}

template< typename T>
image2d<T> operator /(image2d<T> A, image2d<T> B)
{
    image2d<T> C(A.nrows(), A.ncols());
    pixel_wise(A, B, C) | [&] (auto a, auto b, auto &c)
    {
        c = a / b;
    };
    return C;
}


template< typename T>
image2d<T> operator ^(image2d<T> A, image2d<T> B)
{
    image2d<T> C(A.nrows(), A.ncols());
    pixel_wise(A, B, C) | [&] (auto a, auto b, auto &c)
    {
        c = pow(a, b);
    };
    return C;
}
#endif

}
