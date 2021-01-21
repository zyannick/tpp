#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <Eigen/Core>

 
using namespace std;
using namespace Eigen;

namespace tpp
{
	inline void eigen_to_opencv(Eigen::MatrixXd eigen_mat, cv::Mat &cv_mat)
	{
		cv_mat = cv::Mat::zeros(eigen_mat.rows(), eigen_mat.cols(), CV_64FC1);
		for (int row = 0; row < eigen_mat.rows(); row++)
		{
			for (int col = 0; col < eigen_mat.cols(); col++)
			{
				cv_mat.at<double>(row, col) = eigen_mat(row, col);
			}
		}
	}

	inline void eigen_to_opencv_float(Eigen::MatrixXf eigen_mat, cv::Mat &cv_mat)
	{
		cv_mat = cv::Mat::zeros(eigen_mat.rows(), eigen_mat.cols(), CV_64FC1);
		for (int row = 0; row < eigen_mat.rows(); row++)
		{
			for (int col = 0; col < eigen_mat.cols(); col++)
			{
				cv_mat.at<float>(row, col) = eigen_mat(row, col);
			}
		}
	}

	inline void opencv_to_eigen(cv::Mat cv_mat, Eigen::MatrixXd &eigen_mat)
	{
		eigen_mat = MatrixXd::Zero(cv_mat.rows, cv_mat.cols);
		for (int row = 0; row < cv_mat.rows; row++)
		{
			for (int col = 0; col < cv_mat.rows; col++)
			{
				eigen_mat(row, col) = cv_mat.at<double>(row, col);
			}
		}
	}

	inline void opencv_to_eigen_float(cv::Mat cv_mat, Eigen::MatrixXf &eigen_mat)
	{
		eigen_mat = MatrixXf::Zero(cv_mat.rows, cv_mat.cols);
		for (int row = 0; row < cv_mat.rows; row++)
		{
			for (int col = 0; col < cv_mat.rows; col++)
			{
				eigen_mat(row, col) = cv_mat.at<uchar>(row, col);
			}
		}
	}
}
