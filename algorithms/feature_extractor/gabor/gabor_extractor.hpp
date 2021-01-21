#pragma once

//#include "gabor_filter.hh"

#include "gabor_util.hh"

#include "log_gabor.hh"

#include "mono_genic_log_gabor.hh"

#include "phase_congruency_log_gabor.hh"

#include "algorithms/math_functions.hh"

#include <opencv2/core/core.hpp>

#include <opencv2/highgui/highgui.hpp>

namespace tpp
{
	using namespace std;
	using namespace Eigen;

	void log_gabor_extractor(char * file_name, METHOD_LOG_GABOR_EXTRA meth = METHOD_LOG_GABOR_EXTRA::SLIDING_WIND_MAX)
	{
		cv::Mat img_cv = cv::imread(file_name, 0);
		int nrows = img_cv.rows, ncols = img_cv.cols;
		Eigen::MatrixXf img;
		log_gabor t_log_gabor;
		log_gabor_settings t_log_sets;
		set_log_gabor_default_settings(t_log_sets);
		int number_of_filter = t_log_gabor.create_plan(ncols, nrows, t_log_sets);
		opencv_to_eigen_float(img_cv, img);
		t_log_gabor.execut_plan(img);
		std::vector<Eigen::MatrixXf> magnitudes(number_of_filter);
		t_log_gabor.export_gabor_magnitue(magnitudes);
		std::vector<list_vector> interest_points = slide_windows_features_local_maxima1(magnitudes, nrows, ncols, number_of_filter, 10, 10, 4);
		cout << "here " << interest_points.size() << endl;
		for (int i = 0; i < interest_points.size(); i++)
		{
			list_vector ltt = interest_points[i];
			cout << "liste " << ltt.size() << " nb " << i << endl;
			for (auto &l : ltt)
			{
				std::vector<float> val = l;
				cout << "coord " << val[0] << "   " << val[1] << "  " << val[2] << endl;
			}
			cout << endl << endl << endl;
		}
	}
}
