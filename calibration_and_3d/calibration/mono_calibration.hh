#pragma once

#include <iostream>
#include <string.h>
#include <cctype>
#include <stdio.h>
#include <time.h>
#include <ctime>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <set>
#include <list>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include "define_calibrate.hh"
#include "design_objet_board.hh"
#include "save_param.hh"
#include "extract_points.hh"
#include "algorithms/miscellaneous.hh"

 
using namespace std;

namespace tpp
{
	/// Effectue la calibration Mono.
    static bool runCalibration(std::vector<std::vector<cv::Point2f> > imagePoints,
		cv::Size imageSize, cv::Size boardSize, Pattern patternType,
		float squareSize, float aspectRatio,
		int flags, cv::Mat& cameraMatrix, cv::Mat& distCoeffs,
        std::vector<cv::Mat>& rvecs, std::vector<cv::Mat>& tvecs,
        std::vector<float>& reprojErrs,
		double& totalAvgErr)
	{
		cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
		if (flags & cv::CALIB_FIX_ASPECT_RATIO)
			cameraMatrix.at<double>(0, 0) = aspectRatio;

		distCoeffs = cv::Mat::zeros(8, 1, CV_64F);

        std::vector<std::vector<cv::Point3f> > objectPoints(1);
		CalculateObjectsPoints(boardSize, squareSize, objectPoints[0], patternType);

		objectPoints.resize(imagePoints.size(), objectPoints[0]);

		totalAvgErr = calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix,
			distCoeffs, rvecs, tvecs, flags | cv::CALIB_FIX_K4 | cv::CALIB_FIX_K5);
		///*|CALIB_FIX_K3*/|CALIB_FIX_K4|CALIB_FIX_K5);

		bool ok = cv::checkRange(cameraMatrix) && cv::checkRange(distCoeffs);

		return ok;
	}

	static bool runCalibration(/*Settings& s,*/cv::Size boardSize, float squareSize, int flag, Pattern pat, cv::Size& imageSize, cv::Mat& cameraMatrix, cv::Mat& distCoeffs,
        std::vector<std::vector<cv::Point2f> > imagePoints, std::vector<cv::Mat>& rvecs, std::vector<cv::Mat>& tvecs,
        std::vector<float>& reprojErrs, double& totalAvgErr)
	{
		cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
        if (flag & cv::CALIB_FIX_ASPECT_RATIO)
			cameraMatrix.at<double>(0, 0) = 1.0;

		distCoeffs = cv::Mat::zeros(8, 1, CV_64F);

        std::vector<std::vector<cv::Point3f> > objectPoints(1);
		calcBoardCornerPositions(boardSize, squareSize, objectPoints[0], pat);

		objectPoints.resize(imagePoints.size(), objectPoints[0]);

		//Find intrinsic and extrinsic camera parameters
		double rms = cv::calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix,
            distCoeffs, rvecs, tvecs, flag |  cv::CALIB_FIX_K4 | cv::CALIB_FIX_K5,
			cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-9));

		std::cout << "Re-projection error reported by calibrateCamera: " << rms << std::endl;

		bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);

		totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints,
			rvecs, tvecs, cameraMatrix, distCoeffs, reprojErrs);

		return ok;
	}

	bool runCalibrationAndSave(/*Settings& s,*/cv::Size boardSize, float squareSize, int flag, Pattern pat,
		cv::Size imageSize, cv::Mat&  cameraMatrix, cv::Mat& distCoeffs, std::vector<std::vector<cv::Point2f> > imagePoints, string file_name, float aspect_ratio)
	{
        std::vector<cv::Mat> rvecs, tvecs;
        std::vector<float> reprojErrs;
		double totalAvgErr = 0;

		bool ok = runCalibration(boardSize, squareSize, flag, pat, imageSize, cameraMatrix, distCoeffs, imagePoints, rvecs, tvecs,
			reprojErrs, totalAvgErr);
		cout << (ok ? "Calibration succeeded" : "Calibration failed")
			<< ". avg re projection error = " << totalAvgErr;

		if (ok)
			saveCameraParams(file_name, imageSize, boardSize, squareSize, aspect_ratio, flag,
				cameraMatrix, distCoeffs, rvecs, tvecs, reprojErrs, imagePoints, totalAvgErr);
		return ok;
	}

	static
		void mono_calibration(cv::Mat cameraMatrix, cv::Mat distCoeffs,
            std::vector<string> &left_images,
			cv::Mat &R, cv::Mat &T, cv::Mat &E, cv::Mat &F, cv::Mat &Q,
			Mode_To_Revtreive_Files mode, string inputFile = "")
	{
        std::vector<string> imageListCam;
		string inputFilename;
		if (mode == DIRECTORY)
		{
			//getListFilesOfDirectory(imageListCam1, imageListCam2);
		}
		else if (mode == JSFILES)
		{
			if (inputFile.length() == 0)
				inputFilename = std::string("imageList.json");// fichier par défaut
			else
				inputFilename = inputFile;
		}
		bool displayCorners = false;
		bool useCalibrated = true;
		bool showRectified = true;
		cv::Size boardSize, imageSize;
		float squareSize;

		//Mat cameraMatrix[2], distCoeffs[2];
		string outputFilename;

		int flags;
        std::vector<std::vector<cv::Point2f> > imagePoints;
		clock_t prevTimestamp = 0;
        std::vector<string> goodImageList;
		Pattern pattern;
		float aspect_ratio;
		const cv::Scalar RED(0, 0, 255), GREEN(0, 255, 0);
		const char ESC_KEY = 27;

		boardSize.width = 6;
		boardSize.height = 6;
		pattern = CIRCLES_GRID;
		squareSize = 160;
		outputFilename = std::string("out_mono_camera_data.json");
        flags |= cv::CALIB_FIX_ASPECT_RATIO;
        flags |= cv::CALIB_FIX_PRINCIPAL_POINT;
        flags |= cv::CALIB_USE_INTRINSIC_GUESS ;
        flags |= cv::CALIB_ZERO_TANGENT_DIST ;

		int i, j, k, nimages;

		remplirListes(inputFilename, imageListCam);
		nimages = imageListCam.size();

		assert(nimages >= 2 && "The calibration process needs at least 2 images pairs");

		bool extrac_ok;
		int number_points = boardSize.height*boardSize.width;
		int nb_ok = 0;

		for (int i = 0; i < (int)imageListCam.size(); i++)
		{
			cv::Mat view = cv::imread(imageListCam[i], CV_16U);
            std::vector<cv::Point2f> im_point;
			extrac_ok = extrairePointsImage(view, boardSize, im_point, imageListCam[i]);
			if (!extrac_ok)
			{
				cout << "not ok 1" << endl;
				continue;
			}
			//cout << endl << endl << endl;

			assert(imagePoints[i].size() == number_points && "The extraction is not well performed");

			imagePoints[i] = im_point;

			cout << "Image pair number : " << i + 1 << endl;

			goodImageList.push_back(imageListCam[i]);
			nb_ok++;
			imageSize = view.size();
		}

		runCalibrationAndSave(boardSize, squareSize, flags, pattern, imageSize,
			cameraMatrix, distCoeffs, imagePoints, outputFilename, aspect_ratio);
	}

	static
		void mono_calibration(cv::Mat cameraMatrix, cv::Mat distCoeffs,
            std::vector<string> &left_images, cv::Size boardSize,
            float squareSize, Pattern pattern, cv::Size imageSize, std::vector<std::vector<cv::Point2f> > imagePoints, int flags, float aspect_ratio)
	{
		string outputFilename = std::string("out_mono_camera_data.json");
		runCalibrationAndSave(boardSize, squareSize, flags, pattern, imageSize,
			cameraMatrix, distCoeffs, imagePoints, outputFilename, aspect_ratio);
	}
}
