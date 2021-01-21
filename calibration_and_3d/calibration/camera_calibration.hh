#pragma once
#include <Eigen/Core>
#include <iostream>
#include <vector>
#include <opencv2/core.hpp>

#include "algorithms/math_functions.hh"

using namespace Eigen;
using namespace std;


namespace tpp
{
	struct camera_calibration
	{
		VectorXd distorsion_matrix;
		MatrixXf camera_matrix;
	};

	struct stereo_params_eigen
	{
		stereo_params_eigen();
		stereo_params_eigen(MatrixXf distorsion_matrix_left, MatrixXf camera_matrix_left, MatrixXf R_left, MatrixXf P_left, MatrixXf distorsion_matrix_right,
			MatrixXf camera_matrix_right, MatrixXf R_right, MatrixXf P_right, MatrixXf rotation_matrix, MatrixXf rotation_vector, MatrixXf translation_vector,
			MatrixXf F, MatrixXf E, MatrixXf Q) : D1(distorsion_matrix_left), M1(camera_matrix_left), R1(R_left), P1(P_left),
			D2(distorsion_matrix_right), M2(camera_matrix_right), R2(R_right), P2(P_right), R(rotation_matrix),
			rotation_vector(rotation_vector), T(translation_vector), F(F), E(E), Q(Q) {};

		MatrixXf D1;
		MatrixXf M1;
		MatrixXf R1;
		MatrixXf P1;

		MatrixXf D2;
		MatrixXf M2;
		MatrixXf R2;
		MatrixXf P2;

		MatrixXf R;
		MatrixXf rotation_vector;
		MatrixXf T;
		MatrixXf F;
		MatrixXf E;
		MatrixXf Q;
	};

	struct stereo_params_cv
	{
        stereo_params_cv();
		stereo_params_cv(stereo_params_eigen st);
        void retreive_values();

		cv::Mat D1;
		cv::Mat M1;
		cv::Mat R1;
		cv::Mat P1;

		cv::Mat D2;
		cv::Mat M2;
		cv::Mat R2;
		cv::Mat P2;

		cv::Mat R;
		cv::Mat rotation_vector;
		cv::Mat T;
		cv::Mat F;
		cv::Mat E;
		cv::Mat Q;
	};
}

#include "camera_calibration.hpp"
