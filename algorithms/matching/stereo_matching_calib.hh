#pragma once

#include <Eigen/Dense>
#include <Eigen/Core>
#include <list>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "calibration_and_3d/calibration/stereo_calibration.hh"
#include "calibration_and_3d/recontruction3D/recontruction3D.hh"

using namespace Eigen;
using namespace tpp;

namespace tpp{


void init_undistort_rectify_map(stereo_params_cv stereo_params, cv::Mat &cam1map1, cv::Mat &cam1map2,
    cv::Mat &cam2map1, cv::Mat& cam2map2)
{
    cv::Mat left, right; //Create matrices for storing input images

    cv::initUndistortRectifyMap(stereo_params.M1, Mat(),
                            stereo_params.R1, stereo_params.P1, Size(80, 60), CV_16SC2, cam1map1, cam1map2);
    cv::initUndistortRectifyMap(stereo_params.M2, Mat(),
                            stereo_params.R2, stereo_params.P2, Size(80, 60), CV_16SC2, cam2map1, cam2map2);
}

void un_distort_stereo_images(cv::Mat& leftStereoUndistorted, cv::Mat& rightStereoUndistorted, cv::Mat left, cv::Mat right, cv::Mat cam1map1,
    cv::Mat cam1map2, cv::Mat cam2map1, cv::Mat cam2map2)
{
    //Rectify and undistort images
    cv::remap(left, leftStereoUndistorted, cam1map1, cam1map2, INTER_CUBIC);
    cv::remap(right, rightStereoUndistorted, cam2map1, cam2map2, INTER_CUBIC);

}

void matching_stereo_calib()
{
    stereo_params_cv stereo_par;
    stereo_par.retreive_values();

    vector<string> left_images, right_images;
    string inputFilename;
    inputFilename = std::string("./json/depth_4.json");

    remplirListes(inputFilename, left_images, right_images);
    assert(left_images.size() == right_images.size() && "The two listes must have the same size");
    int nimages = left_images.size();

    cv::Mat cam1map1, cam1map2;
    cv::Mat cam2map1, cam2map2;

    init_undistort_rectify_map(stereo_par, cam1map1, cam1map2,
                               cam2map1, cam2map2);

    cout << "nimages " << nimages << endl;

    for (int ster = 0; ster < nimages; ster++)
    {
        cv::Mat left_view_16 = imread(left_images[ster], CV_16U);
        cv::Mat left_view, left_view_rect;

        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(left_view_16, &minVal, &maxVal, &minLoc, &maxLoc);
        left_view_16.convertTo(left_view, cv::CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));

        cv::Mat right_view_16 = cv::imread(right_images[ster], CV_16U);
        cv::Mat right_view, right_view_rect;
        cv::minMaxLoc(right_view_16, &minVal, &maxVal, &minLoc, &maxLoc);
        right_view_16.convertTo(right_view, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));

        cv::un_distort_stereo_images(left_view_rect, right_view_rect, left_view, right_view, cam1map1, cam1map2,
                                 cam2map1, cam2map2);

        cv::Mat left_r = cv::Mat::zeros(left_view.rows,left_view.cols,CV_8U);

        cv::resize(left_view_rect, left_r, left_r.size(), 0, 0, INTER_AREA);

        cv::imwrite("t.bmp", left_view);

        cv::imwrite("left.bmp", left_r);
        cv::imwrite("right.bmp", right_view_rect);
        return;
    }
}


void init_und(MatrixXf camera_matrix, VectorXf dist_coeff, MatrixXf R, MatrixXf new_camera_matrix, int nrows, int ncols, MatrixXf map1, MatrixXf map2)
{
    float cx,cy,fx,fy;
    cx = camera_matrix(0,2);
    cy = camera_matrix(1,2);
    fx = camera_matrix(0,0);
    fy = camera_matrix(1,1);

    float cx_,cy_,fx_,fy_;
    cx_ = new_camera_matrix(0,2);
    cy_ = new_camera_matrix(1,2);
    fx_ = new_camera_matrix(0,0);
    fy_ = new_camera_matrix(1,1);

    float k1,k2,p1,p2,k3;
    k1 = dist_coeff(0);
    k2 = dist_coeff(0);
    p1 = dist_coeff(0);
    p2 = dist_coeff(0);
    k3 = dist_coeff(0);


    for(auto row =0 ; row < nrows; row++)
    {
        for(auto col = 0 ; col < ncols ; col++)
        {
            //float x = y = 0;
            //x = (col - camera_matrix())
        }
    }

}

}
