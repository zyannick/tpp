#pragma once

#include "log_gabor_filter_bank.hpp"
#include <string>

// from https://github.com/chvillap/phase-congruency-features/blob/master/src/filter_bank/log_gabor_filter_bank.cpp

using namespace std;

namespace tpp
{
	log_gabor_filter_bank::log_gabor_filter_bank(std::string    filename_prefix,
		triple<size_t> sizes,
		size_t         num_scales,
		size_t         num_azimuths,
		size_t         num_elevations,
		float          max_frequency,
		float          mult_factor,
		float          frequency_ratio,
		float          angular_ratio,
		float          lowpass_order,
		float          lowpass_cutoff,
		bool           uniform_sampling)
	{
		filter_filename_prefix = filename_prefix;
		filter_sizes = sizes;
		filter_num_scales = num_scales;
		filter_num_azimuths = num_azimuths;
		filter_num_elevations = num_elevations;
		filter_max_frequency = max_frequency;
		filter_mult_factor = mult_factor;
		filter_frequency_ratio = frequency_ratio;
		filter_angular_ratio = angular_ratio;
		filter_lowpass_order = lowpass_order;
		filter_lowpass_cutoff = lowpass_cutoff;
		filter_uniform_sampling = uniform_sampling;

		filter_num_azimuths_per_elevation = new size_t[filter_num_elevations]();
		filter_num_azimuths_per_elevation[0] = filter_num_azimuths;

		// With uniform sampling, every elevation has the same number of azimuth
		// angles. With nonuniform sampling, we reduce the number of azimuths as
		// the elevation angle increases, according to the formula below.
		if (filter_uniform_sampling) {
			for (size_t e = 1; e < filter_num_elevations; ++e)
				filter_num_azimuths_per_elevation[e] = filter_num_azimuths;
		}
		else {
			if (filter_num_elevations > 1)
				filter_num_azimuths_per_elevation[filter_num_elevations - 1] = 1;

			for (size_t e = 1; e < filter_num_elevations - 1; ++e) {
				float n = filter_num_azimuths * cos(e * M_PI_2 / (filter_num_elevations - 1));
				filter_num_azimuths_per_elevation[e] = (size_t)round(n) * 2;
			}
		}
	}

	void log_gabor_filter_bank::compute()
	{
		/*
		* TODO:
		* Parallelize this method by dividing the computation of the nested loops
		* below in multiple threads. Each filter is computed independently of the
		* others, so the performance could be greatly improved with that.
		*
		* PS: choose between parallelizing this method or create_filter() method.
		* Parallelizing both is probably not a good idea (thread overhead).
		*/

		// Variation in the elevation angle.
		float dtheta = (filter_num_elevations == 1) ? 0.0 : M_PI_2 / (filter_num_elevations - 1);

		for (size_t e = 0; e < filter_num_elevations; ++e) {
			// Variation in the azimuth angle.
			float dphi = 0.0;
			if (filter_num_azimuths > 1) {
				dphi = (e == 0) ? M_PI / filter_num_azimuths_per_elevation[e] : M_PI / filter_num_azimuths_per_elevation[e] * 2;
			}
			// Central elevation angle.
			float theta0 = e * dtheta;

			for (size_t a = 0; a < filter_num_azimuths_per_elevation[e]; ++a) {
				// Central azimuth angle.
				float phi0 = a * dphi;

				for (size_t s = 0; s < filter_num_scales; ++s) {
					// Central frequency.
					float freq0 = filter_max_frequency / pow(filter_mult_factor, s);
					// Create the filter for the current frequency and orientation.
					float *filter = create_filter(freq0, phi0, theta0);
					delete[] filter;
				}
			}
		}
	}

	float* log_gabor_filter_bank::create_filter(float freq0, float phi0, float theta0)
	{
		double EPSILON = 1e-9;
		/*
		* TODO:
		* Parallelize this method by dividing the computation of the nested loops
		* below in multiple threads. Each voxel is computed independently of the
		* others, so the performance could be greatly improved with that.
		*/

		// Angular standard deviation.
		const float std_alpha = M_PI / filter_num_azimuths / filter_angular_ratio;

		// Trigonometric constants.
		const float sin_theta = sin(theta0);
		const float cos_theta = cos(theta0);
		const float cos_theta_cos_phi = cos_theta * cos(phi0);
		const float cos_theta_sin_phi = cos_theta * sin(phi0);

		// Allocate the filter data array.
		float *filter = new float[filter_sizes[0] * filter_sizes[1] * filter_sizes[2]]();

		// Iterate through the frequency domain coordinates.
		for (size_t i = 0, z = 0; z < filter_sizes[2]; ++z) {
			float w = (filter_sizes[2] == 1) ? 0.0 : 0.5 - (float)z / filter_sizes[2];

			for (size_t y = 0; y < filter_sizes[1]; ++y) {
				float v = (filter_sizes[1] == 1) ? 0.0 : 0.5 - (float)y / filter_sizes[1];

				for (size_t x = 0; x < filter_sizes[0]; ++x, ++i) {
					float u = (filter_sizes[0] == 1) ? 0.0 : -0.5 + (float)x / filter_sizes[0];

					// Get the frequency value.
					float freq = sqrt(u*u + v*v + w*w);

					if (freq < EPSILON)
						filter[i] = 0.0;
					else {
						// Project the sample point in the cartesian space.
						float uu = u * cos_theta_cos_phi;
						float vv = v * cos_theta_sin_phi;
						float ww = w * sin_theta;

						// Angular distance between the central frequency point
						// and the current sample point.
						float alpha = acos((uu + vv + ww) / (freq + EPSILON));

						// Angular and radial components in the frequency domain.
						float spread = exp(-0.5 * (alpha / std_alpha) *  (alpha / std_alpha));
						float val = (log(freq / freq0) / log(filter_frequency_ratio));
						float radius = exp(-0.5 * val * val);

						// // Angular distance between the central frequency point
						// // and the current sample.
						// float alpha = fabs(acos((uu + vv + ww) / (freq + EPSILON)));
						// alpha = std::min(alpha * m_num_azimuths/2.0, M_PI);

						// // Angular and radial components in the frequency domain.
						// float spread = 0.5 * (1 + cos(alpha));
						// float radius = exp(-0.5 * sqr(log(freq / freq0) /
						//                               log(m_frequency_ratio)));

						// Apply a Butterworth low-pass filter.
						radius /= (1 + pow((freq / filter_lowpass_cutoff), filter_lowpass_order * 2));

						// Combine the radial and angular spread components.
						filter[i] = radius * spread;
					}
				}
			}
		}
		return filter;
	}

	float*
		log_gabor_filter_bank::
		get_filter(size_t scale, size_t azimuth, size_t elevation)
	{
		// Try to read an existing filter from file.
		float *filter /*= read_filter(scale, azimuth, elevation)*/;

		return filter;
	}
}