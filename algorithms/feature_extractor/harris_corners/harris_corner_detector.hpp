#pragma once

#include "harris_corners_detector.hh"

namespace tpp
{
	 
	using namespace std;

	cv::RNG rng_harris(12345);

	harris_detector_settings::harris_detector_settings(int tresh, int max_tresh, int block_size, int aperture_size, double k)
	{
		this->thresh = thresh;
		this->max_tresh = max_tresh;
		this->block_size = block_size;
		this->aperture_size = aperture_size;
		this->k = k;
		this->initialized = true;
	}

	harris_detector_settings::harris_detector_settings()
	{
	}

	void default_initialization_harris_settings(harris_detector_settings &st)
	{
		st.thresh = 150;
		st.max_tresh = 255;
		st.block_size = 2;
		st.aperture_size = 3;
		st.k = 0.04;
		st.initialized = true;
	}

    void extract_corners_harris(cv::Mat src_gray, std::vector<cv::Point2f>& corners, harris_detector_settings st)
	{
        //Mat src_gray;
        //cvtColor(src, src_gray, cv::COLOR_RGB2GRAY);
		cv::Mat dst, dst_norm, dst_norm_scaled;
        //cout << "cornerHarris " << endl;
		cornerHarris(src_gray, dst, st.block_size, st.aperture_size, st.k, cv::BORDER_DEFAULT);
        //cout << "here" << endl;
		cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
		cv::convertScaleAbs(dst_norm, dst_norm_scaled);
		int r = 4;
        
        for (int j = 0; j < dst_norm.rows; j++)
		{
			for (int i = 0; i < dst_norm.cols; i++)
			{
				if ((int)dst_norm.at<float>(j, i) > st.thresh)
				{
                    corners.push_back(cv::Point2f(i, j));
                    //circle(src, Point(i, j), 5, Scalar(rng_harris.uniform(0, 255), rng_harris.uniform(0, 255), rng_harris.uniform(0, 255)), -1, 8, 0);
				}
			}
		}
        
        /*imwrite("harris_result.jpg", src);*/
	}

	void extract_corners_harris_(cv::Mat img_in, std::vector<cv::Point2f>& corners, harris_detector_settings st, int i, int j)
	{
		double minVal, maxVal;
		cv::Point minLoc, maxLoc;
		std::string NomFichier;
		cv::minMaxLoc(img_in, &minVal, &maxVal, &minLoc, &maxLoc);
		cv::Mat src;
		img_in.convertTo(src, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
		cv::cvtColor(src, src, cv::COLOR_GRAY2BGR);
		cv::Mat src_gray;
		cv::cvtColor(src, src_gray, cv::COLOR_RGB2GRAY);
		cv::Mat dst, dst_norm, dst_norm_scaled;
		cornerHarris(src_gray, dst, st.block_size, st.aperture_size, st.k, cv::BORDER_DEFAULT);
		normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
		convertScaleAbs(dst_norm, dst_norm_scaled);
		int r = 4;
		for (int j = 0; j < dst_norm.rows; j++)
		{
			for (int i = 0; i < dst_norm.cols; i++)
			{
				if ((int)dst_norm.at<float>(j, i) > st.thresh)
				{
					corners.push_back(cv::Point2f(i, j));
					circle(src, cv::Point(i, j), 5, cv::Scalar(rng_harris.uniform(0, 255), rng_harris.uniform(0, 255), rng_harris.uniform(0, 255)), -1, 8, 0);
				}
			}
		}
		std::string ch = std::string("").append(std::string("harris_result")).append(std::to_string(i)).append(std::to_string(j)).append(std::string(".bmp"));
		imwrite(ch, src);
	}
}
