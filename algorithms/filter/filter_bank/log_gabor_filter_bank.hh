#pragma once
#define _USE_MATH_DEFINES

#include <cstddef>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <assert.h>

using namespace std;

namespace tpp
{
	template <class T>
	using triple = std::array<T, 3>;

	struct log_gabor_filter_bank
	{
		log_gabor_filter_bank(std::string    filename_prefix,
			triple<size_t> sizes,
			size_t         num_scales = 4,
			size_t         num_azimuths = 6,
			size_t         num_elevations = 3,
			float          max_frequency = 1. / 3,
			float          mult_factor = 2.1,
			float          frequency_ratio = 0.55,
			float          angular_ratio = 1.2,
			float          lowpass_order = 15.0,
			float          lowpass_cutoff = 0.45,
			bool           uniform_sampling = false);

		~log_gabor_filter_bank();

		float* get_filter(size_t scale, size_t azimuth = 0, size_t elevation = 0);

		void compute();

		size_t get_num_azimuths(size_t elevation) const {
			assert(elevation < filter_num_elevations);
			return filter_num_azimuths_per_elevation[elevation];
		}

		size_t get_num_azimuths() const {
			return filter_num_azimuths;
		}

		/**
		* @brief Calculates the total number of filter orientations according to
		* the numbers of azimuth and elevation angles and the filter sampling.
		*
		* @returns The total number of filter orientations.
		*/
		size_t get_num_orientations() const {
			size_t result = 0;
			for (size_t e = 0; e < filter_num_elevations; ++e)
				result += filter_num_azimuths_per_elevation[e];
			return result;
		}

		float* create_filter(float freq0, float phi0, float theta0);

		std::string    filter_filename_prefix;
		triple<size_t> filter_sizes;
		size_t         filter_num_scales;
		size_t         filter_num_azimuths;
		size_t        *filter_num_azimuths_per_elevation;
		size_t         filter_num_elevations;
		float          filter_max_frequency;
		float          filter_mult_factor;
		float          filter_frequency_ratio;
		float          filter_angular_ratio;
		float          filter_lowpass_order;
		float          filter_lowpass_cutoff;
		bool           filter_uniform_sampling;
	};
}

#include "log_gabor_filter_bank.hpp"