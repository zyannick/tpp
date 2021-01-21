#pragma once

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

 
using namespace std;

namespace tpp
{
	struct shi_tomasi_settings
	{
		shi_tomasi_settings();
		shi_tomasi_settings(int max_corners, int max_corners_possible, int block_size, double quality_level, double min_distance, bool use_harris_detector, double k);
		int max_corners;
		int max_corners_possible;
		int block_size;
		double quality_level;
		double min_distance;
		bool use_harris_detector;
		double k;
		bool initialized = false;
	};

	void default_initialization_shi_tomasi(shi_tomasi_settings &st);

	void extract_shi_tomasi_corners(cv::Mat src_gray, std::vector<cv::Point2f>& corners, shi_tomasi_settings st);
}

#include "shi_tomasi_extractor.hpp"