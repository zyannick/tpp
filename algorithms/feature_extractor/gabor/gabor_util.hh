#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include "fftw3.h"
#include <Eigen/Core>

using namespace Eigen;

#pragma warning (disable: 4018 4244 4305)

#ifdef _MSC_VER
#define NOMINMAX
#include <minmax.h>
#endif

namespace tpp
{

    double sqrt_log_4 = sqrt(log(4));
	typedef std::list<std::vector<float>> list_vector;

	//////////////////////////////////////////////////////////////////////////
	///		auxiliary functions for 2D FFT
	//////////////////////////////////////////////////////////////////////////

	/*!
	transform an 2D mat to a 1D array \n
	input store by row-major order, in should be outside initialized
	*/
	void eigen_matrix_to_float_array(MatrixXf imga, float* in);

	/// transform a reduced hermit symmetry matrix to a full 2D array, for FFTW
	void hermite_to_array(fftwf_complex* out, fftwf_complex* ret, int nw, int nh);

	/// 2D FFTShift on complex data
	void _2d_fft_shift(fftwf_complex* out, int nw, int nh);

	enum METHOD_LOG_GABOR_EXTRA { SLIDING_WIND_MAX, SLIDING_WIND_MOMENTS, SLIDING_WIND_MAX_PCA };

	//////////////////////////////////////////////////////////////////////////
	///		statistics feature for given Gabor response
	//////////////////////////////////////////////////////////////////////////

	/*!
	calculating the first two order moment for a given array, i.e, mean and standard deviation
	@param points -- input array
	@param sz -- the size of the input array
	@param vmu -- first statistics, mean value
	@param vstd -- second statistics, std deviation value
	*/
	void compute_moment_stats(VectorXf points, int sz, float& vmu, float& vstd);

	void slide_windows_features_moments(VectorXf &vecFea, float** pGabMag, int nh, int nw, int nFB, int xblocks, int yblocks);

	void slide_windows_features_local_maxima(VectorXf &vecFea, float** pGabMag, int nh, int nw, int nFB, int xblocks, int yblocks);

	std::vector<list_vector> slide_windows_features_local_maxima1(std::vector<MatrixXf> pGabMag, int nh, int nw, int nFB, int xblocks, int yblocks, int wind);

	void magnitude_to_image(VectorXf pMag, cv::Mat img);
}

#include "gabor_util.hpp"
