#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

using namespace std;

namespace tpp
{
	static void CalculateObjectsPoints(const cv::Size &boardSize,
		const float &squareSize, vector<cv::Point3f>& corners, Pattern patternType = CHESSBOARD)
	{
		corners.resize(0);
		switch (patternType)
		{
		case CHESSBOARD:
		case CIRCLES_GRID:
			for (int j = boardSize.height - 1; j >= 0; j--)
				for (int i = 0; i < boardSize.width; i++)
					corners.push_back(cv::Point3f(float(i*squareSize), float(j*squareSize), 0));//plan de la grille z=0 .
			break;

		case ASYMMETRIC_CIRCLES_GRID:
			for (int i = 0; i < boardSize.height; i++)
				for (int j = 0; j < boardSize.width; j++)
					corners.push_back(cv::Point3f(float((2 * j + i % 2)*squareSize),
						float(i*squareSize), 0));
			break;

		default:
			CV_Error(cv::Error::StsBadArg, "Unknown pattern type\n");
		}
	}

	static void CalculateObjectsPoints_Stereo(const cv::Size &boardSize,
		const float &squareSize, vector<vector<cv::Point3f> >& corners, int nimages, Pattern patternType = CHESSBOARD)
	{
		switch (patternType)
		{
		case CHESSBOARD:
		case CIRCLES_GRID:
			for (int i = 0; i < nimages; i++)
				for (int j = 0; j < boardSize.height; j++)
					for (int k = 0; k < boardSize.width; k++)
						corners[i].push_back(cv::Point3f(float(k*squareSize), float(j*squareSize), 0));//plan de la grille z=0 .
			break;

		case ASYMMETRIC_CIRCLES_GRID:
			for (int k = 0; k < nimages; k++)
				for (int i = 0; i < boardSize.height; i++)
					for (int j = 0; j < boardSize.width; j++)
						corners[k].push_back(cv::Point3f(float((2 * j + i % 2)*squareSize),
							float(i*squareSize), 0));
			break;

		default:
			CV_Error(cv::Error::StsBadArg, "Unknown pattern type\n");
		}
	}

	static void calcBoardCornerPositions(cv::Size boardSize, float squareSize, vector<cv::Point3f>& corners,
		Pattern patternType /*= Settings::CHESSBOARD*/)
	{
		corners.clear();

		switch (patternType)
		{
		case CHESSBOARD:
		case CIRCLES_GRID:
			for (int i = 0; i < boardSize.height; ++i)
				for (int j = 0; j < boardSize.width; ++j)
					corners.push_back(cv::Point3f(float(j*squareSize), float(i*squareSize), 0));
			break;

		case ASYMMETRIC_CIRCLES_GRID:
			for (int i = 0; i < boardSize.height; i++)
				for (int j = 0; j < boardSize.width; j++)
					corners.push_back(cv::Point3f(float((2 * j + i % 2)*squareSize), float(i*squareSize), 0));
			break;
		default:
			break;
		}
	}

	static double computeReprojectionErrors(const vector<vector<cv::Point3f> >& objectPoints,
		const vector<vector<cv::Point2f> >& imagePoints,
		const vector<cv::Mat>& rvecs, const vector<cv::Mat>& tvecs,
		const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs,
		vector<float>& perViewErrors)
	{
		vector<cv::Point2f> imagePoints2;
		int i, totalPoints = 0;
		double totalErr = 0, err;
		perViewErrors.resize(objectPoints.size());

		for (i = 0; i < (int)objectPoints.size(); ++i)
		{
			cv::projectPoints(cv::Mat(objectPoints[i]), rvecs[i], tvecs[i], cameraMatrix,
				distCoeffs, imagePoints2);
            err = cv::norm(cv::Mat(imagePoints[i]), cv::Mat(imagePoints2), cv::NORM_L2);

			int n = (int)objectPoints[i].size();
			perViewErrors[i] = (float)std::sqrt(err*err / n);
			totalErr += err*err;
			totalPoints += n;
		}

		return std::sqrt(totalErr / totalPoints);
	}
}
