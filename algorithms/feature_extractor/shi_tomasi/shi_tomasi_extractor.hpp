#pragma once

#include "shi_tomasi_extractor.hh"

namespace tpp
{
	 
	using namespace std;

	cv::RNG rng_st(12345);

	inline
		void default_initialization_shi_tomasi(shi_tomasi_settings &st)
	{
        st.block_size = 3;
		st.quality_level = 0.01;
        st.min_distance = 1;
		st.use_harris_detector = false;
		st.k = 0.04;
        st.max_corners = 10000;
        st.max_corners_possible = 1000000;
		st.initialized = true;
	}

	shi_tomasi_settings::shi_tomasi_settings()
	{
	}

	shi_tomasi_settings::shi_tomasi_settings(int max_corners, int max_corners_possible, int block_size, double quality_level, double min_distance, bool use_harris_detector, double k)
	{
		this->block_size = block_size;
		this->quality_level = quality_level;
		this->min_distance = min_distance;
		this->use_harris_detector = use_harris_detector;
		this->k = k;
		this->max_corners = (max_corners < 1) ? 1 : max_corners;
		this->max_corners_possible = max_corners_possible;
		this->initialized = true;
	}

    void extract_shi_tomasi_corners(cv::Mat src, std::vector<cv::Point2f>& corners, shi_tomasi_settings st)
	{
		cv::Mat src_gray;
		cv::cvtColor(src, src_gray, cv::COLOR_RGB2GRAY);
		/// Apply corner detection
		goodFeaturesToTrack(src_gray,
			corners,
			st.max_corners,
			st.quality_level,
			st.min_distance,
			cv::Mat(),
			st.block_size,
			st.use_harris_detector,
			st.k);
        int r = 1;
        for (int i = 0; i < corners.size(); i++)
		{
            src.at<cv::Vec3b>(corners[i]) = cv::Vec3b(rng_st.uniform(100, 255), rng_st.uniform(100, 255),rng_st.uniform(100, 255));
            //circle(src, corners[i], r, Scalar(rng_st.uniform(0, 255), rng_st.uniform(0, 255),rng_st.uniform(0, 255)), -1, 8, 0);
		}

        imwrite("shi_tomasi_example.bmp", src);
		return;
	}
}
