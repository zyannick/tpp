#pragma once

#define _USE_MATH_DEFINES
#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
//#include <unsupported\Eigen\CXX11\src\Tensor\Tensor.h>
#include "fftw3.h"
#include "gabor_util.hh"

#define  FFTW_IN_PLACE
#pragma comment(lib, "fftw3f")

using namespace Eigen;

namespace tpp
{
    //float sqrt_log_4 = sqrt(log(4));

	struct log_gabor_settings
	{
		int number_of_scale;			/// Number of wavelet scales, default 3 scales
		int number_of_orientations;		/// Number of filter orientations, default 8 orientations
		int smallest_scale_filter;		/// Wavelength of smallest scale filter, at least 3, default 4 pixels

							// kapa = exp(-0.25 beta sqrt(2 ln2) ),
							// beta: bandwidth in octave
							// kapa = 0.745 for beta = 1, kapa=0.555 for beta = 2
		float band_width_octave;		/// band width in octave, in the range of 0.5~4 octave, default 1 octave
		float angular_inter;	/// Ratio of angular interval between filter orientations, default 1.7
		bool is_initialized;
	};

	/// set default parameters for Log-Gabor filters
	inline void set_log_gabor_default_settings(log_gabor_settings& gabor_param)
	{
		gabor_param.number_of_scale = 3;
		gabor_param.number_of_orientations = 8;
		gabor_param.smallest_scale_filter = 4;	// at least 3 pixel, 4 is better

		gabor_param.band_width_octave = 1.0f;	//
		gabor_param.angular_inter = 1.7f;	// best value
		gabor_param.is_initialized = true;
	}

	class log_gabor
	{
	public:
		/// constructor
		log_gabor();

		/// destructor
		~log_gabor();

	public:
		/*!
		create a shared plan for Log-Gabor filter \n
		the input 2D signals or images of the same size can use one plan forever

		@param nImgWidth -- width of the image
		@param nImgHeight -- width of the image
		@param gab_param -- parameters of the Log-Gabor filters
		@return number of filter-banks
		*/
		int create_plan(int width, int height, log_gabor_settings gab_param);

		/*!
		execute the plan for a given matrix
		@param img -- input of the OpenCV CvMat format
		*/
		void execut_plan(MatrixXf img);

		/// destroy the plan
		void destroyPlan();

		/// export pointer of gaborMagnitude (i.e., pointer of each row of m_gabMag)
		void export_gabor_magnitue(std::vector<MatrixXf>& gaborMag);

	private:
		/// pre-computing kernel of Gabor filters
		void precompute_gabor_kernel(int nw, int nh);

	private:
		/// Gabor parameters
		float scaling_factor;  /// Scaling factor between successive filters
		float sigma_octave_band;  /// Gaussian sigma for octave band, smaller for larger band width

							/// parameters for the Log-Gabor filters
		log_gabor_settings log_gabor_sets;

		/// Gabor magnitude and phase
		MatrixXf  magnitudes;		/// magnitude
		MatrixXf  phases;		/// phase

											//////////////////////////////////////////////////////////////////////////
											/// the following parameters are for shared plan execution
											//////////////////////////////////////////////////////////////////////////
		int height;	/// height
		int width;	/// width
		int number_of_filter_banks;	/// number of filter bank
		int m_nType;

		/// kernels of Gabor filters
		MatrixXf gabor_kernel;

		/// convolution by FFT in the freq-domain
		/// in-place transformation, both before and after FFT
		fftwf_complex *m_imgfftConv;

#ifndef FFTW_IN_PLACE
		/// output of Log-Gabor transformation for out-of-place transformation
		fftwf_complex *m_gaborOut;
#endif

		/// the FFTW plan for inverse FFT
		fftwf_plan m_invplan;

		// image FFT variables
		float *m_imgin;
		fftwf_complex *m_fftwOut;
		fftwf_complex *m_imgfft;
		fftwf_plan m_fwplan;
	};
}

#include "log_gabor.hpp"
