#pragma once
#define _USE_MATH_DEFINES

#include <Eigen/Core>
#include <complex>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>
#include <algorithm>
#include <array>
#include <iostream>
#include "algorithms/math_functions.hh"
#include <fftw3.h>

using namespace Eigen;
using namespace std;

namespace tpp
{
	bool
		fp_equal(float n1, float n2, float maxerror)
	{
		// Try a simple equality test first.
		if (n1 == n2)
			return true;
		// Then try a "fuzzy" comparison using the absolute error.
		return fabs(n1 - n2) < maxerror;
	}

	double sqr(double x)
	{
		return x*x;
	}

	Vector3d cart2sph(Vector3d cart)
	{
		const float x = cart[0];
		const float y = cart[1];
		const float z = cart[2];
		const float rho = sqrt(sqr(x) + sqr(y) + sqr(z));
		const float phi = atan2(y, x);
		const float theta = atan2(z, sqrt(sqr(x) + sqr(y)));

		Vector3d sph = { rho, phi, theta };
		return sph;
	}

	Vector3d sph2cart(Vector3d sph)
	{
		const float rho = sph[0];
		const float phi = sph[1];
		const float theta = sph[2];
		const float x = rho * cos(theta) * cos(phi);
		const float y = rho * cos(theta) * sin(phi);
		const float z = rho * sin(theta);

		Vector3d cart = { x, y, z };
		return cart;
	}

	float
		median(float *array, size_t size)
	{
		assert(array != NULL);

		// Copy the data to an auxiliary vector.
		std::vector<float> aux;
		aux.assign(array, array + size);

		// Find the element in the middle of the data array.
		// Notice that this function rearranges the vector. That's why we need to
		// copy the original array first.
		size_t n = aux.size() / 2;
		std::nth_element(aux.begin(), aux.begin() + n, aux.end());

		// If the data size is odd, we have the median element.
		if (aux.size() % 2)
			return aux[n];

		// Otherwise, we compute the average of the nth and (n-1)th elements.
		std::nth_element(aux.begin(), aux.begin() + n - 1, aux.end());
		return 0.5 * (aux[n] + aux[n - 1]);
	}
}
