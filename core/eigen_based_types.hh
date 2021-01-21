#pragma once
#define _USE_MATH_DEFINES

#include <Eigen/Core>
#include <iostream>
#include <complex>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace Eigen;
using namespace std;

double EPSILON = 1e-9;

typedef std::complex<double> complex_double;

typedef Eigen::MatrixXcd image_complex;

// This
typedef Eigen::Vector3d TVector;

//typedef Tensor<double, 3> _3D_image_d;

//typedef Tensor<std::complex<double>, 3> _3D_image_complexe;

typedef Array<bool, Dynamic, 1> ArrayXb;
