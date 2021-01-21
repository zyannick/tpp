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

 
using namespace std;

namespace tpp
{
	/// Enregistre les parametres mono
	static void saveCameraParams(const string& filename,
		cv::Size imageSize, cv::Size boardSize,
		float squareSize, float aspectRatio, int flags,
		const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs,
		const std::vector<cv::Mat>& rvecs, const std::vector<cv::Mat>& tvecs,
		const std::vector<float>& reprojErrs,
		const std::vector<std::vector<cv::Point2f> >& imagePoints,
		double totalAvgErr)
	{
		cv::FileStorage fs(filename, cv::FileStorage::WRITE);//le format est determine automatiquement a partir de l extension

		fs << "distortion" << distCoeffs;
		//debut calcul des extrinsics
		if (!rvecs.empty() && !tvecs.empty())
		{
			CV_Assert(rvecs[0].type() == tvecs[0].type());
			cv::Mat bigmat((int)rvecs.size(), 6, rvecs[0].type());
			for (int i = 0; i < (int)rvecs.size(); i++)
			{
				cv::Mat r = bigmat(cv::Range(i, i + 1), cv::Range(0, 3));
				cv::Mat t = bigmat(cv::Range(i, i + 1), cv::Range(3, 6));

				CV_Assert(rvecs[i].rows == 3 && rvecs[i].cols == 1);
				CV_Assert(tvecs[i].rows == 3 && tvecs[i].cols == 1);
				//*.t() is MatExpr (not Mat) so we can use assignment operator
				r = rvecs[i].t();
				t = tvecs[i].t();
			}//fin calcul des extrinsics
			fs << "extrinsic" << bigmat;// On ecrit les extrinsics seulement si la caloibration les a trouves
		}
		fs << "intrinsic" << cameraMatrix;
		fs << "avg_reprojection_error" << totalAvgErr;
	}

	/// Enregistre les paramètres des caméras.
	static void saveStereoParams(const string& filename,
		const cv::Size &imageSize, const cv::Mat& cameraMatrix1, const cv::Mat& cameraMatrix2,
		const cv::Mat& distCoeffs1, const cv::Mat& distCoeffs2,
		const cv::Mat R, const cv::Mat T, const cv::Mat E, const cv::Mat F,
		double totalAvgErr)
	{
		//Calcul des vecteurs de rotation
		vector<double> rotation;
		cv::Rodrigues(R, rotation);

		for (int i = 0; i < rotation.size(); i++)
		{
			rotation[i] = rotation[i] * 180 / CV_PI;
			//printf("\n%lf\n",rotation[i]);
		}

		/*//extraction de la matrice essentielle
		cv::Mat Rext1;
		vector<double> Text,Rext;
		cv::recoverPose(E, imagePoints1, imagePoints2, Rext1, Text, 1.0);
		cv::Rodrigues(Rext1, Rext);
		for (int i = 0; i < Rext.size(); i++)
		{
		Rext[i] = Rext[i] * 180 / CV_PI;
		}*/

		cv::FileStorage fs(filename, cv::FileStorage::WRITE);//le format est determine automatiquement a partir de l extension

		fs << "distortion camera 1" << distCoeffs1;
		fs << "distortion camera 2" << distCoeffs2;
		fs << "intrinsic camera 1" << cameraMatrix1;
		fs << "intrinsic camera 2" << cameraMatrix2;
		fs << "matrice de rotation entre les 2 cameras" << R;
		fs << "vecteur de rotation entre les 2 cameras en degres dans le sens trigo" << rotation;
		fs << "vecteur de translation entre les 2 cameras" << T;
		fs << "matrice essentielle" << E;
		//fs << "vecteur de rotation extrinseque en degres dans le sens trigo" << Rext;
		//fs << "vecteur de translation extrinseque" << Text;
		fs << "matrice fondamentale" << F;
		fs << "avg_reprojection_error" << totalAvgErr;
	}
}
