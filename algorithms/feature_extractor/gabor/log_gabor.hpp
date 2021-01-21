#pragma once

#include <algorithm>
#include "log_gabor.hh"
#include "opencv2/highgui.hpp"
#include "algorithms/miscellaneous.hh"
//#include <unsupported/Eigen/MatrixFunctions>

namespace tpp
{
	log_gabor::log_gabor()
	{
		number_of_filter_banks = 0;
		height = 0;
		width = 0;
	}

	log_gabor::~log_gabor()
	{
		destroyPlan();
	}

	// Pre-compute Gabor Kernels
	void log_gabor::precompute_gabor_kernel(int nw, int nh)
	{
		int i, j, idx;
		int nw2 = nw / 2 + 1;
		int nh2 = nh / 2 + 1;
		int sz = nw * nh;

		VectorXf sinTheta, cosTheta, radius, spread;
		sinTheta = VectorXf::Zero(sz);
		cosTheta = VectorXf::Zero(sz);
		radius = VectorXf::Zero(sz);
		spread = VectorXf::Zero(sz);

		// temporal variable
		float x, y, theta;
		int negnw2 = -nw / 2;
		int negnh2 = -nh / 2;
		float invnw = 1.0 / nw;
		float invnh = 1.0 / nh;

		// in-place computing
		int offset;
		for (j = 0; j < nh; j++) // rows
		{
			offset = j*nw;
			y = (negnh2 + j) * invnh;
			for (i = 0; i < nw; i++) // cols
			{
				x = (negnw2 + i) * invnw;
				theta = atan2(-y, x);

				idx = i + offset;
				sinTheta[idx] = sin(theta);
				cosTheta[idx] = cos(theta);
				radius[idx] = 0.5 * log(x*x + y*y);
			}
		}
		// setting the center to be 1, log(1) = 0
		radius[nw / 2 + (nh / 2) * nw] = 0;

		//////////////////////////////////////////////////////////////////////////
		float angl, ds, dc, dtheta, cosAngl, sinAngl;
		float val, logWav, wavLength;

		float invThetaSigma2 = M_PI / (log_gabor_sets.number_of_orientations * log_gabor_sets.angular_inter);
		invThetaSigma2 = -0.5 / (invThetaSigma2 * invThetaSigma2);
		float invLogSigmaOn2 = log(sigma_octave_band);
		invLogSigmaOn2 = -0.5 / (invLogSigmaOn2 * invLogSigmaOn2);

		VectorXf pKernel;
		gabor_kernel.resize(number_of_filter_banks, sz);
		for (int i = 0; i < log_gabor_sets.number_of_orientations; i++) // orientation
		{
			wavLength = log_gabor_sets.smallest_scale_filter;

			// computing spread
			angl = (i*M_PI) / log_gabor_sets.number_of_orientations;
			cosAngl = cos(angl);
			sinAngl = sin(angl);
			for (int k = 0; k < sz; k++)
			{
				ds = sinTheta[k] * cosAngl - cosTheta[k] * sinAngl;	// Difference in sine.
				dc = cosTheta[k] * cosAngl + sinTheta[k] * sinAngl; // Difference in cosine
				dtheta = atan2(ds, dc);	 // Absolute angular distance.
				spread[k] = exp(dtheta * dtheta * invThetaSigma2);
			}

			for (int j = 0; j < log_gabor_sets.number_of_scale; j++) // scale
			{
				// logGabor
				logWav = log(wavLength);
				for (int k = 0; k < sz; k++)
				{
					// most time consuming point
					val = radius[k] + logWav;
					gabor_kernel(i*log_gabor_sets.number_of_scale + j, k) = exp(val * val * invLogSigmaOn2);
				}
				gabor_kernel(i*log_gabor_sets.number_of_scale + j, (nw2 - 1) + (nh2 - 1) * nw) = 0;

				// logGabor filter
				for (int k = 0; k < sz; k++)
				{
					gabor_kernel(i*log_gabor_sets.number_of_scale + j, k) = gabor_kernel(i*log_gabor_sets.number_of_scale + j, k) * spread[k];
				}

				// calculate Wavelength of the next level filter
				wavLength = wavLength * scaling_factor;
			}
		}
	}

	// create plan for shared size image, allocate some memory
	int log_gabor::create_plan(int nImgWidth, int nImgHeight, log_gabor_settings gab_param /* =NULL */)
	{
		width = nImgWidth;
		height = nImgHeight;
		int sz = width * height;

		if (gab_param.is_initialized)
			log_gabor_sets = gab_param;
		else
			set_log_gabor_default_settings(log_gabor_sets);

		number_of_filter_banks = log_gabor_sets.number_of_orientations * log_gabor_sets.number_of_scale;

		// within 1~4 octave
		if (log_gabor_sets.band_width_octave < 0.5 || log_gabor_sets.band_width_octave > 4)
		{
			sigma_octave_band = 0.745;
			float m_rMultCoeff = 0.5 * M_PI;
		}
		else
		{
			sigma_octave_band = exp(-0.25 * log_gabor_sets.band_width_octave * sqrt(2 * log(2.0)));
			if (log_gabor_sets.band_width_octave >= 1.0)
				scaling_factor = 0.5 * M_PI * log_gabor_sets.band_width_octave;
			else
				scaling_factor = 0.5 * M_PI * (0.745 / sigma_octave_band);
		}

		// pre-computing radius and the polar angle of all pixels: m_radius, m_spread
		precompute_gabor_kernel(width, height);

		// memory for inverse 2D FFT
		m_imgfftConv = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * sz);

#ifndef FFTW_IN_PLACE
		m_gaborOut = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * sz);
		m_invplan = fftwf_plan_dft_2d(m_nh, m_nw, m_imgfftConv, m_gaborOut, FFTW_BACKWARD, FFTW_ESTIMATE);
#else
		m_invplan = fftwf_plan_dft_2d(height, width, m_imgfftConv, m_imgfftConv, FFTW_BACKWARD, FFTW_ESTIMATE);
#endif

		// output feature memory
		magnitudes.resize(number_of_filter_banks, sz);

		// variables for image FFT
		// the most important is the FFTW output memory layout
		int nw2 = width / 2 + 1;
		int fftwSz = height * nw2;
        //m_imgin = (float *)cvAlloc(sizeof(float) * sz);

		m_fftwOut = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * fftwSz);
		m_imgfft = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * sz);

		// special note: the input should be row major
		m_fwplan = fftwf_plan_dft_r2c_2d(height, width, m_imgin, m_fftwOut, FFTW_ESTIMATE);

		return number_of_filter_banks;
	}

	void log_gabor::destroyPlan()
	{
		if (height * width > 0)
		{
			// release memory
			// free the gabor feature memory

#ifndef FFTW_IN_PLACE
			fftwf_free(m_gaborOut);
#endif

			fftwf_free(m_fftwOut);
			fftwf_free(m_imgfft);
			fftwf_free(m_imgfftConv);

            //cvFree(&m_imgin);
			fftwf_destroy_plan(m_invplan);
			fftwf_destroy_plan(m_fwplan);

			number_of_filter_banks = 0;
			height = 0;
			width = 0;
		}
	}

	// now full compatible with the MATLAB version log-Gabor filter
	void log_gabor::execut_plan(MatrixXf img)
	{
		int nw = img.cols();
		int nh = img.rows();
		int sz = nw * nh;

		if (nw != width || nh != height)
			return;

		//S1: 2D FFT transform the input image to frequency domain
		// initialize the input data after creating the plan
		eigen_matrix_to_float_array(img, m_imgin);

		// execute the FFT plan
		fftwf_execute(m_fwplan);

		// transform from FFTW Hermitian symmetry to real two dimensional format
		hermite_to_array(m_fftwOut, m_imgfft, width, height);

		// shift FFT image with the same phase of the Gabor filter
		_2d_fft_shift(m_imgfft, width, height);

		// S2: start computing Log-Gabor filter in frequency domain
		//real *pKernel;
		for (int i = 0; i < number_of_filter_banks; i++)
		{
			//pKernel = &gabor_kernel[i][0];

			// do the convolution:
			// note in the filter-bank, the order is orientation x scale (continued store same orientation)
			for (int k = 0; k < sz; k++)
			{
				m_imgfftConv[k][0] = m_imgfft[k][0] * gabor_kernel(i, k);
				m_imgfftConv[k][1] = m_imgfft[k][1] * gabor_kernel(i, k);
			}

			// back FFT transform, and save the result in m_gaborOut
			fftwf_execute(m_invplan);

			// computing the magnitude
			//pKernel = &magnitudes[i][0];

			for (int k = 0; k < sz; k++)
			{
#ifndef FFTW_IN_PLACE
				magnitudes(i, k) = sqrt(m_gaborOut[k][0] * m_gaborOut[k][0] + m_gaborOut[k][1] * m_gaborOut[k][1]);
#else
				magnitudes(i, k) = sqrt(m_imgfftConv[k][0] * m_imgfftConv[k][0] + m_imgfftConv[k][1] * m_imgfftConv[k][1]);
#endif
			}
		} // end i
	}

	void log_gabor::export_gabor_magnitue(std::vector<MatrixXf> & gaborMag)
	{
		int k = 0;
		for (int i = 0; i < number_of_filter_banks; i++)
		{
			k = 0;
			(gaborMag[i]) = MatrixXf::Zero(height, width);
			for (int row = 0; row < height; row++)
			{
				for (int col = 0; col < width; col++)
				{
					k = row * height + col;
					(gaborMag[i])(row, col) = magnitudes(i, k);
				}
			}
			//gaborMag.row(i) = magnitudes.row(i);
		}
	}

}
