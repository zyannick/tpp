#pragma once

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

 
using namespace std;

namespace tpp
{
	struct harris_detector_settings
	{
		harris_detector_settings(int tresh, int max_tresh, int block_size, int aperture_size, double k);
		harris_detector_settings();
		int thresh;
		int max_tresh;
		int block_size;
		int aperture_size;
		double k;
		bool initialized = false;
	};

	void default_initialization_harris_settings(harris_detector_settings &st);

	void extract_corners_harris(cv::Mat src_gray, std::vector<cv::Point2f>& corners, harris_detector_settings st);
}

#include "harris_corner_detector.hpp"