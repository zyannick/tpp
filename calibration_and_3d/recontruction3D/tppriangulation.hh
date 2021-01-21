#pragma once

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"

#include <Eigen/Core>
#include "algorithms/miscellaneous.hh"
#include "calibration_and_3d/calibration/camera_calibration.hh"
#include "plane_reconstruction.hh"

#include <iostream>
#include <fstream>
#include <string>
using namespace std;
 
//using namespace cv::sfm;
using namespace Eigen;

#include <time.h>

namespace tpp
{
	cv::Mat triangulate_Linear_LS(cv::Mat mat_P_l, cv::Mat mat_P_r, cv::Mat warped_back_l, cv::Mat warped_back_r)
	{
		cv::Mat A(4, 3, CV_64FC1), b(4, 1, CV_64FC1), X(3, 1, CV_64FC1), X_homogeneous(4, 1, CV_64FC1), W(1, 1, CV_64FC1);
		W.at<double>(0, 0) = 1.0;
		A.at<double>(0, 0) = (warped_back_l.at<double>(0, 0) / warped_back_l.at<double>(2, 0))*mat_P_l.at<double>(2, 0) - mat_P_l.at<double>(0, 0);
		A.at<double>(0, 1) = (warped_back_l.at<double>(0, 0) / warped_back_l.at<double>(2, 0))*mat_P_l.at<double>(2, 1) - mat_P_l.at<double>(0, 1);
		A.at<double>(0, 2) = (warped_back_l.at<double>(0, 0) / warped_back_l.at<double>(2, 0))*mat_P_l.at<double>(2, 2) - mat_P_l.at<double>(0, 2);
		A.at<double>(1, 0) = (warped_back_l.at<double>(1, 0) / warped_back_l.at<double>(2, 0))*mat_P_l.at<double>(2, 0) - mat_P_l.at<double>(1, 0);
		A.at<double>(1, 1) = (warped_back_l.at<double>(1, 0) / warped_back_l.at<double>(2, 0))*mat_P_l.at<double>(2, 1) - mat_P_l.at<double>(1, 1);
		A.at<double>(1, 2) = (warped_back_l.at<double>(1, 0) / warped_back_l.at<double>(2, 0))*mat_P_l.at<double>(2, 2) - mat_P_l.at<double>(1, 2);
		A.at<double>(2, 0) = (warped_back_r.at<double>(0, 0) / warped_back_r.at<double>(2, 0))*mat_P_r.at<double>(2, 0) - mat_P_r.at<double>(0, 0);
		A.at<double>(2, 1) = (warped_back_r.at<double>(0, 0) / warped_back_r.at<double>(2, 0))*mat_P_r.at<double>(2, 1) - mat_P_r.at<double>(0, 1);
		A.at<double>(2, 2) = (warped_back_r.at<double>(0, 0) / warped_back_r.at<double>(2, 0))*mat_P_r.at<double>(2, 2) - mat_P_r.at<double>(0, 2);
		A.at<double>(3, 0) = (warped_back_r.at<double>(1, 0) / warped_back_r.at<double>(2, 0))*mat_P_r.at<double>(2, 0) - mat_P_r.at<double>(1, 0);
		A.at<double>(3, 1) = (warped_back_r.at<double>(1, 0) / warped_back_r.at<double>(2, 0))*mat_P_r.at<double>(2, 1) - mat_P_r.at<double>(1, 1);
		A.at<double>(3, 2) = (warped_back_r.at<double>(1, 0) / warped_back_r.at<double>(2, 0))*mat_P_r.at<double>(2, 2) - mat_P_r.at<double>(1, 2);
		b.at<double>(0, 0) = -((warped_back_l.at<double>(0, 0) / warped_back_l.at<double>(2, 0))*mat_P_l.at<double>(2, 3) - mat_P_l.at<double>(0, 3));
		b.at<double>(1, 0) = -((warped_back_l.at<double>(1, 0) / warped_back_l.at<double>(2, 0))*mat_P_l.at<double>(2, 3) - mat_P_l.at<double>(1, 3));
		b.at<double>(2, 0) = -((warped_back_r.at<double>(0, 0) / warped_back_r.at<double>(2, 0))*mat_P_r.at<double>(2, 3) - mat_P_r.at<double>(0, 3));
		b.at<double>(3, 0) = -((warped_back_r.at<double>(1, 0) / warped_back_r.at<double>(2, 0))*mat_P_r.at<double>(2, 3) - mat_P_r.at<double>(1, 3));
		cv::solve(A, b, X, cv::DECOMP_SVD);
		vconcat(X, W, X_homogeneous);
		return X_homogeneous;
	}

	/**
	From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
	*/
	cv::Mat_<double> LinearLSTriangulation(cv::Point3d u,       //homogenous image point (u,v,1)
		cv::Matx34d P,       //camera 1 matrix
		cv::Point3d u1,      //homogenous image point in 2nd camera
		cv::Matx34d P1       //camera 2 matrix
	)
	{
		//build matrix A for homogenous equation system Ax = 0
		//assume X = (x,y,z,1), for Linear-LS method
		//which turns it into a AX = B system, where A is 4x3, X is 3x1 and B is 4x1
		cv::Matx43d A(u.x*P(2, 0) - P(0, 0), u.x*P(2, 1) - P(0, 1), u.x*P(2, 2) - P(0, 2),
			u.y*P(2, 0) - P(1, 0), u.y*P(2, 1) - P(1, 1), u.y*P(2, 2) - P(1, 2),
			u1.x*P1(2, 0) - P1(0, 0), u1.x*P1(2, 1) - P1(0, 1), u1.x*P1(2, 2) - P1(0, 2),
			u1.y*P1(2, 0) - P1(1, 0), u1.y*P1(2, 1) - P1(1, 1), u1.y*P1(2, 2) - P1(1, 2)
		);
		cv::Mat_<double> B = (cv::Mat_<double>(4, 1) << -(u.x*P(2, 3) - P(0, 3)),
			-(u.y*P(2, 3) - P(1, 3)),
			-(u1.x*P1(2, 3) - P1(0, 3)),
			-(u1.y*P1(2, 3) - P1(1, 3)));

		cv::Mat_<double> X;
		cv::solve(A, B, X, cv::DECOMP_SVD);

		return X;
	}

	/**
	From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
	*/
	cv::Mat_<double> IterativeLinearLSTriangulation(cv::Point3d u,    //homogenous image point (u,v,1)
		cv::Matx34d P,          //camera 1 matrix
		cv::Point3d u1,         //homogenous image point in 2nd camera
		cv::Matx34d P1          //camera 2 matrix
	) {
		double wi = 1, wi1 = 1;
		cv::Mat_<double> X(4, 1);
		int i;

		cv::Mat_<double> X_ = LinearLSTriangulation(u, P, u1, P1);

		X(0) = X_(0);
		X(1) = X_(1);
		X(2) = X_(2);
		X(3) = 1.0;

		for (i = 0; i < 10; i++) { //Hartley suggests 10 iterations at most

            if(i > 0)
            {
                X(0) = X_(0);
                X(1) = X_(1);
                X(2) = X_(2);
                X(3) = 1.0;

                //cout << "Itereation " << i << "  X(0) " << X(0) << "  X(1) " << X(1) << " X(2) " << X(2) << " X(3) " << X(3) << endl;
            }

			double p2x = cv::Mat_<double>(cv::Mat_<double>(P).row(2)*X)(0);
			double p2x1 = cv::Mat_<double>(cv::Mat_<double>(P1).row(2)*X)(0);

			//cout << "fabsf(wi - p2x)  " << fabsf(wi - p2x) << "  fabsf(wi1 - p2x1)  " << fabsf(wi1 - p2x1) << endl ;

            //breaking point]
            //cout << "Itereation 1 " << i << endl;
			if (fabsf(wi - p2x) <= 1e-9 && fabsf(wi1 - p2x1) <= 1e-9) break;

			wi = p2x;
			wi1 = p2x1;

			//reweight equations and solve
			cv::Matx43d A((u.x*P(2, 0) - P(0, 0)) / wi, (u.x*P(2, 1) - P(0, 1)) / wi, (u.x*P(2, 2) - P(0, 2)) / wi,
				(u.y*P(2, 0) - P(1, 0)) / wi, (u.y*P(2, 1) - P(1, 1)) / wi, (u.y*P(2, 2) - P(1, 2)) / wi,
				(u1.x*P1(2, 0) - P1(0, 0)) / wi1, (u1.x*P1(2, 1) - P1(0, 1)) / wi1, (u1.x*P1(2, 2) - P1(0, 2)) / wi1,
				(u1.y*P1(2, 0) - P1(1, 0)) / wi1, (u1.y*P1(2, 1) - P1(1, 1)) / wi1, (u1.y*P1(2, 2) - P1(1, 2)) / wi1
			);
            //cout << "Itereation 2 " << i << endl;
			cv::Mat_<double> B = (cv::Mat_<double>(4, 1) << -(u.x*P(2, 3) - P(0, 3)) / wi,
				-(u.y*P(2, 3) - P(1, 3)) / wi,
				-(u1.x*P1(2, 3) - P1(0, 3)) / wi1,
				-(u1.y*P1(2, 3) - P1(1, 3)) / wi1
				);
            //cout << "Itereation 3 " << i << endl;

			cv::solve(A, B, X_, cv::DECOMP_SVD);

            //cout << "Itereation 4 " << i << endl;

            //cout << "Itereation " << i << "  X(0) " << X(0) << "  X(1) " << X(1) << " X(2) " << X(2) << " X(3) " << X(3) << endl;

		}

		return X;
	}
}
