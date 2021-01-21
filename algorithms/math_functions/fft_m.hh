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
#include "fftw3.h"

using namespace Eigen;
using namespace std;

/*

void FFT(Tensor<std::complex<double>, 3> M, Vector3d sizes, bool backward = false)
{
	const size_t dimensions = 3;

	const char WISDOM_FILENAME_FORWARD[32] = "wisdom_fftwf_forward.txt";
	const char WISDOM_FILENAME_BACKWARD[32] = "wisdom_fftwf_backward.txt";

	fftwf_plan plan;
	FILE *wisdom_file = NULL;

	// Open the correct wisdom file according to the desired transform.
	if (backward)
		wisdom_file = fopen(WISDOM_FILENAME_BACKWARD, "r");
	else
		wisdom_file = fopen(WISDOM_FILENAME_FORWARD, "r");

	// Import FFTW plan settings from wisdom file if possible.
	// This will save a lot of runtime in this FFT computation.
	if (wisdom_file) {
		fftwf_import_wisdom_from_file(wisdom_file);
		fclose(wisdom_file);
	}

	// Reverse the order of the sizes array (so the size in Z axis is the first
	// element).
	int ft_sizes[3];
	for (size_t d = 0; d < dimensions; ++d)
		ft_sizes[d] = static_cast<int>(sizes[dimensions - d - 1]);

	// Create the correct FFTW plan according to the desired transform.
	if (backward) {
		plan = fftwf_plan_dft(dimensions, ft_sizes, M, M, FFTW_BACKWARD,
			FFTW_ESTIMATE);
	}
	else {
		plan = fftwf_plan_dft(dimensions, ft_sizes, M, M, FFTW_FORWARD,
			FFTW_ESTIMATE);
	}

	// Export the FFTW plan settings to the wisdom file.
	// This will save a lot of runtime in future FFT computations.
	if (wisdom_file) {
		fftwf_export_wisdom_to_file(wisdom_file);
		fclose(wisdom_file);
	}

	// Compute the transform and clean up.
	fftwf_execute(plan);
	fftwf_destroy_plan(plan);

	// The normalization step is applied only in backward FFT.
	if (backward) {
		size_t total_size = 1;
		for (size_t d = 0; d < dimensions; ++d)
			total_size *= sizes[d];

		for (size_t i = 0; i < total_size; ++i) {
			//M[i][0] /= total_size;
			//M[i][1] /= total_size;
		}
	}
}
*/
