#pragma once


#define _USE_MATH_DEFINES
#include <cstddef>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <new>
#include <assert.h>
#include <fftw3.h>
#include <Eigen/Core>
#include <complex>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>
#include <algorithm>
#include <array>

#include "algorithms/filter/filter_bank/log_gabor_filter_bank.hh"
#include "algorithms/math_functions.hh"
#include "core.hpp"



namespace tpp
{



	struct phase_congruency
	{
		phase_congruency(std::string            filename_prefix,
			Tensor<double, 3>                 input_image,
			std::vector<log_gabor_filter_bank> filter_bank,
			Vector3d        sizes,
			ArrayXb                  input_mask,
			float                  noise_threshold = -1.0,
			float                  noise_std = 3.0,
			float                  sigmoid_gain = 10.0,
			float                  sigmoid_cutoff = 0.5);

		~phase_congruency();

		void compute();

		/**     * @brief Computes the DC-shifted forward FFT of the input image.
     *
     * @param[out] target Preallocated complex image to compute.
     *
     * @return The resulting Fourier transform as a complex image.
     */

		void compute_shifted_FFT_(Tensor<std::complex<double>, 3> img_comp);

		/**
		* @brief Applies a single log-Gabor filter to a complex image in the
		* frequency domain.
		*
		* @param[out] f_output  Preallocated output complex image.
		* @param[in]  f_input   Preallocated input complex image.
		* @param[in]  scale     Filter scale parameter.
		* @param[in]  azimuth   Filter azimuth parameter.
		* @param[in]  elevation Filter elevation parameter.
		*/
		void compute_filtering(Tensor<std::complex<double>, 3> f_output,
			Tensor<std::complex<double>, 3>  f_input,
			size_t         scale,
			size_t         azimuth,
			size_t         elevation);

		/**
		* @brief Automatically estimates the noise energy threshold from the
		* amplitudes of filter responses.
		*
		* @param[in] sum_amplitude Array of sums of filter response amplitudes
		* accumulated over all the scales of the bank of filters.
		*
		* @return The estimated noise energy threshold.
		*/
		float estimate_noise_threshold(float *sum_amplitude);

		/**
		* @brief Applies an energy weighting function to suppress responses that
		* locally occur only to a few frequency components.
		*
		* @param[in] energy        Local energy value.
		* @param[in] sum_amplitude Local summed filter response amplitude.
		* @param[in] max_amplitude Local maximum filter response amplitude.
		*
		* @return The weighted local energy value, as a positive real number.
		*/
		float apply_energy_weighting(float energy,
			float sum_amplitude,
			float max_amplitude);


		std::string            m_filename_prefix;
		std::vector<log_gabor_filter_bank> m_filter_bank;
		Vector3d        m_sizes;
		Tensor<double, 3>        m_input_image;
		ArrayXb                  m_input_mask;
		float                  m_noise_threshold;
		float                  m_noise_std;
		float                  m_sigmoid_gain;
		float                  m_sigmoid_cutoff;



	};

}
