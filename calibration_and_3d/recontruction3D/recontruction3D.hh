#pragma once

#define CERES_FOUND true

//#include <opencv2/sfm.hpp>
//#include <opencv2/viz.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"

#include <Eigen/Core>
#include "algorithms/miscellaneous.hh"
#include "calibration_and_3d/calibration/camera_calibration.hh"
#include "plane_reconstruction.hh"
#include "tppriangulation.hh"

#include <iostream>
#include <fstream>
#include <string>
using namespace std;
 
//using namespace cv::sfm;
using namespace Eigen;

#include <time.h>

namespace tpp
{
/**/
struct plane
{
    plane() : A(0), B(0), C(0) , D(0) {}
    plane(double _A,double _B,double _C,double _D) : A(_A), B(_A), C(_C) , D(_D) {}
    double A, B, C , D;
};

plane null_plane()
{
    return plane();
}

void get_far_point()
{
}

void get_3D_points_from_points_correspondences(cv::Mat left_cam, cv::Mat right_cam, cv::Mat cam_left_points,
    cv::Mat cam_right_points, cv::Mat &points3D)
{
    assert(left_cam.size() == right_cam.size() && "Wrong cameras matrices size");
    assert(cam_left_points.size() == cam_right_points.size() && "The number of points is not the same");
    cv::triangulatePoints(left_cam, right_cam, cam_left_points, cam_right_points, points3D);
}

void points_3D_reconstruction(cv::Mat M1, cv::Mat M2, cv::Mat T, std::vector < std::vector<cv::Point2f>> image_points_left,
                              std::vector<std::vector<cv::Point2f>> image_points_right, int nimages)
{
    cv::FileStorage fs("3D_point_output.yml", cv::FileStorage::WRITE);
    for (auto i = 0; i < nimages; i++)
    {
        cout << i + 1 << "th image pair " << endl;
        int nb_point = (image_points_left[i]).size();
        cv::Mat cam_left_points = from_vector_to_matrix(image_points_left[i]);
        cv::Mat cam_right_points = from_vector_to_matrix(image_points_right[i]);
        cv::Mat points3D(1, nb_point, CV_64FC4);
        cv::Mat left_cam(3, 4, CV_64FC1);
        cv::Mat right_cam(3, 4, CV_64FC1);
        for (int y = 0; y < 3; y++)
        {
            for (int x = 0; x < 3; x++)
            {
                left_cam.at<double>(y, x) = M1.at<double>(y, x);
                right_cam.at<double>(y, x) = M2.at<double>(y, x);
            }
            left_cam.at<double>(y, 3) = T.at<double>(y, 0);
            right_cam.at<double>(y, 3) = T.at<double>(y, 0);
        }
        get_3D_points_from_points_correspondences(left_cam, right_cam, cam_left_points, cam_right_points, points3D);
        std::cout << "taille 3D points :   rows --> " << points3D.rows << "       cols--> " << points3D.cols << endl;
        std::cout << "type " << points3D.type() << " channel " << points3D.channels() << endl;
        for (int x = 0; x < points3D.cols; x++)
        {
            for (int y = 0; y < points3D.rows; y++)
            {
                cout << points3D.at<double>(y, x) << "   ";
            }
            cout << endl << endl << endl;
        }
    }
    fs.release();
    system("PAUSE");
}

void distance_3D_point_to_plane(double A, double B, double D, cv::Mat points3D)
{
    double C = -1;

    double sqrtc = sqrt(pow(A, 2) + pow(B, 2) + pow(C, 2));

    for (int col = 0; col < points3D.cols; col++)
    {
        double Xval, Yval, Zval;
        Xval = points3D.at<double>(0, col);
        Yval = points3D.at<double>(1, col);
        Zval = points3D.at<double>(2, col);
        double val = Xval*A + Yval*B - Zval + D;
        double distance = fabs(val) / sqrtc;

        cout << "distance " << distance << endl;
    }
}

void distance_3D_point_to_plane(double A, double B,double C, double D, cv::Mat points3D)
{

    double sqrtc = sqrt(pow(A, 2) + pow(B, 2) + pow(C, 2));

    for (int col = 0; col < points3D.cols; col++)
    {
        double Xval, Yval, Zval;
        Xval = points3D.at<double>(0, col);
        Yval = points3D.at<double>(1, col);
        Zval = points3D.at<double>(2, col);
        double val = Xval*A + Yval*B - Zval + D;
        double distance = fabs(val) / sqrtc;

        cout << "distance " << distance << endl;
    }
}

void save_3D_points(std::vector<cv::Mat> vect_points3D_)
{
    ofstream myZ;
    myZ.open("pointsZwd.txt");
    ofstream myX;
    myX.open("pointsXwd.txt");
    ofstream myY;
    myY.open("pointsYwd.txt");

    for (int i = 0; i < vect_points3D_.size(); i++)
    {
        myX << " X = [ ";
        myY << " Y = [ ";
        myZ << " Z = [ ";

        for (int col = 0; col < vect_points3D_[i].cols; col++)
        {
            double X, Y, Z, W;
            X = vect_points3D_[i].at<double>(0, col);
            Y = vect_points3D_[i].at<double>(1, col);
            Z = vect_points3D_[i].at<double>(2, col);

            myX << X << "  ";
            myY << Y << "  ";
            myZ << Z << "  ";
        }

        myX << " ]; \n";
        myY << " ]; \n";
        myZ << " ]; \n";
    }
}

void save_3D_points(cv::Mat vect_points3D_)
{
    ofstream myZ;
    myZ.open("pointsZwd.txt");
    ofstream myX;
    myX.open("pointsXwd.txt");
    ofstream myY;
    myY.open("pointsYwd.txt");

    myX << " X = [ ";
    myY << " Y = [ ";
    myZ << " Z = [ ";

    for (int col = 0; col < vect_points3D_.cols; col++)
    {
        double X, Y, Z, W;
        X = vect_points3D_.at<double>(0, col);
        Y = vect_points3D_.at<double>(1, col);
        Z = vect_points3D_.at<double>(2, col);

        myX << X << "  ";
        myY << Y << "  ";
        myZ << Z << "  ";
    }

    myX << " ]; \n";
    myY << " ]; \n";
    myZ << " ]; \n";

}


void save_3D_points(cv::Mat vect_points3D_ , int cp)
{
    ofstream myZ;
    myZ.open("pointsZwd.txt");
    ofstream myX;
    myX.open("pointsXwd.txt");
    ofstream myY;
    myY.open("pointsYwd.txt");



    for (int col = 0; col < vect_points3D_.cols; col++)
    {
        if(col %cp ==0)
        {
            myX << " X = [ ";
            myY << " Y = [ ";
            myZ << " Z = [ ";
        }

        double X, Y, Z, W;
        X = vect_points3D_.at<double>(0, col);
        Y = vect_points3D_.at<double>(1, col);
        Z = vect_points3D_.at<double>(2, col);

        myX << X << "  ";
        myY << Y << "  ";
        myZ << Z << "  ";

        if(col%cp ==(cp-1))
        {
            myX << " ]; \n";
            myY << " ]; \n";
            myZ << " ]; \n";

        }
    }


}

void points_3D_reconstruction_list_rectified(cv::Mat F, cv::Mat M1, cv::Mat M2, cv::Mat P1, cv::Mat P2, cv::Mat R1, cv::Mat R2, cv::Mat T, cv::Mat distCoeffs1, cv::Mat distCoeffs2,
                                             std::vector < std::vector<cv::Point2f>> image_points_left,
                                             std::vector<std::vector<cv::Point2f>> image_points_right, int nimages, cv::Size imageSize)
{
    ofstream myZ;
    myZ.open("pointsZwd.txt");
    ofstream myX;
    myX.open("pointsXwd.txt");
    ofstream myY;
    myY.open("pointsYwd.txt");
    ofstream myP;
    myP.open("plane.txt");
    int nb_point;
    for (auto index_img = 0; index_img < nimages; index_img++)
    {
        cout << index_img + 1 << "th image pair " << endl;
        myX << " X = [ ";
        myY << " Y = [ ";
        myZ << " Z = [ ";
        cv::Mat cam_left_points;
        cv::Mat cam_right_points;
        int nb_point;
        nb_point = (image_points_left[index_img]).size();
        cam_left_points = cv::Mat::zeros(1, nb_point, CV_64FC2);
        cam_right_points = cv::Mat::zeros(1, nb_point, CV_64FC2);
        from_vector_to_matrix(image_points_left[index_img], cam_left_points);
        from_vector_to_matrix(image_points_right[index_img], cam_right_points);
        cv::Mat cam_left_points_undistorsed;
        cv::Mat cam_right_points_undistorsed;
        cv::Mat cam_left_points_corrected_matches;
        cv::Mat cam_right_points_corrected_matches;
        cv::undistortPoints(cam_left_points, cam_left_points_undistorsed, M1, distCoeffs1, R1, P1);
        cv::undistortPoints(cam_right_points, cam_right_points_undistorsed, M2, distCoeffs2, R2, P2);
        correctMatches(F, cam_left_points_undistorsed, cam_right_points_undistorsed, cam_left_points_corrected_matches, cam_right_points_corrected_matches);
        cv::Mat points3D_(4, nb_point, CV_64FC1);
        cv::Size im_size = cam_left_points.size();
        cv::Mat matPoints(nb_point, 3, CV_64FC1);
        cv::Mat matZ(nb_point, 1, CV_64FC1);
        cv::Mat matVal(nb_point, 1, CV_64FC1);
        cv::Mat res(3, 1, CV_64FC1);

        for (int idx = 0; idx < nb_point; idx++)
        {
            cv::Mat_<double> Xx(4, 1);
            cv::Point3d point_left;
            point_left.x = cam_left_points_corrected_matches.at<cv::Vec2d>(0, idx)[0];
            point_left.y = cam_left_points_corrected_matches.at<cv::Vec2d>(0, idx)[1];
            point_left.z = 1;

            cv::Point3d point_right;
            point_right.x = cam_right_points_corrected_matches.at<cv::Vec2d>(0, idx)[0];
            point_right.y = cam_right_points_corrected_matches.at<cv::Vec2d>(0, idx)[1];
            point_right.z = 1;
            Xx = IterativeLinearLSTriangulation(point_left, P1, point_right, P2);

            points3D_.at<double>(0, idx) = Xx.at<double>(0, 0);
            points3D_.at<double>(1, idx) = Xx.at<double>(1, 0);
            points3D_.at<double>(2, idx) = Xx.at<double>(2, 0);
            points3D_.at<double>(3, idx) = Xx.at<double>(3, 0);
        }

        for (int col = 0; col < points3D_.cols; col++)
        {
            double X, Y, Z, W;
            X = points3D_.at<double>(0, col);
            Y = points3D_.at<double>(1, col);
            Z = points3D_.at<double>(2, col);
            W = points3D_.at<double>(3, col);

            matPoints.at<double>(col, 0) = X;
            matPoints.at<double>(col, 1) = Y;
            matPoints.at<double>(col, 2) = W;
            matZ.at<double>(col, 0) = Z;

            myX << X << "  ";
            myY << Y << "  ";
            myZ << Z << "  ";
        }

        myX << " ]; \n";
        myY << " ]; \n";
        myZ << " ]; \n";

        cv::solve(matPoints, matZ, res, cv::DECOMP_SVD );
        double A, B, C;
        A = res.at<double>(0, 0);
        B = res.at<double>(1, 0);
        C = res.at<double>(2, 0);
        myP << " a = " << A << ";" << " b = " << B << ";" << " c = " << -1 << ";" << " d = " << C << "; \n";
    }
    system("PAUSE");
}

//This function take only one point in two pair images and triangulate them
void triangulate_point_to_3D_corrected_match(cv::Mat F, cv::Mat M1, cv::Mat M2, cv::Mat P1, cv::Mat P2, cv::Mat R1, cv::Mat R2, cv::Mat T, cv::Mat distCoeffs1, cv::Mat distCoeffs2,
    cv::Point2f image_points_left,
    cv::Point2f image_points_right, cv::Mat& points3D_)
{


    cv::Mat cam_left_points;
    cv::Mat cam_right_points;
    int nb_point = 1;

    cam_left_points = cv::Mat::zeros(1, nb_point, CV_64FC2);
    cam_left_points.at<cv::Vec2d>(0, 0)[0] = image_points_left.x;
    cam_left_points.at<cv::Vec2d>(0, 0)[1] = image_points_left.y;

    cam_right_points = cv::Mat::zeros(1, nb_point, CV_64FC2);
    cam_right_points.at<cv::Vec2d>(0, 0)[0] = image_points_right.x;
    cam_right_points.at<cv::Vec2d>(0, 0)[1] = image_points_right.y;

    cv::Mat cam_left_points_undistorsed;
    cv::Mat cam_right_points_undistorsed;
    cv::Mat cam_left_points_corrected_matches;
    cv::Mat cam_right_points_corrected_matches;

    cv::undistortPoints(cam_left_points, cam_left_points_undistorsed, M1, distCoeffs1, R1, P1);

    cv::undistortPoints(cam_right_points, cam_right_points_undistorsed, M2, distCoeffs2, R2, P2);

    //cout << "triangulate_point_to_3D_corrected_match" << endl;
    //cout << F.cols << endl;
    //cout << cam_left_points_undistorsed.cols << endl;
    //cout << cam_left_points_undistorsed.rows << endl;

    cv::correctMatches(F, cam_left_points_undistorsed, cam_right_points_undistorsed, cam_left_points_corrected_matches, cam_right_points_corrected_matches);

    points3D_ = cv::Mat::zeros(4, nb_point, CV_64FC1);

   // cout << "boucle " << endl;

    for (int idx = 0; idx < nb_point; idx++)
    {
        cv::Mat_<double> Xx(4, 1);
        cv::Point3d point_left;
        point_left.x = cam_left_points_undistorsed.at<cv::Vec2d>(0, idx)[0];
        point_left.y = cam_left_points_undistorsed.at<cv::Vec2d>(0, idx)[1];
        point_left.z = 1;
        cv::Point3d point_right;

        point_right.x = cam_right_points_undistorsed.at<cv::Vec2d>(0, idx)[0];
        point_right.y = cam_right_points_undistorsed.at<cv::Vec2d>(0, idx)[1];
        point_right.z = 1;
        Xx = IterativeLinearLSTriangulation(point_left, P1, point_right, P2);

        //cv::triangulatePoints(point_left, point_right, cam_left_points, cam_right_points, points3D_);

        points3D_.at<double>(0, idx) = Xx.at<double>(0, 0);
        points3D_.at<double>(1, idx) = Xx.at<double>(1, 0);
        points3D_.at<double>(2, idx) = Xx.at<double>(2, 0);
        points3D_.at<double>(3, idx) = Xx.at<double>(3, 0);
    }
}

//This function takes a list of points coming from the same pair images and triangulates them
void triangulate_plan_to_3D(cv::Mat F, cv::Mat M1, cv::Mat M2, cv::Mat P1, cv::Mat P2, cv::Mat R1, cv::Mat R2, cv::Mat T, cv::Mat distCoeffs1, cv::Mat distCoeffs2,
                            std::vector<cv::Point2f> image_points_left,
                            std::vector<cv::Point2f> image_points_right, cv::Mat& points3D_,
                            FIT_PLANE_CLOUD_POINT fit_plane_cloud_point, plane & pl)
{
    cv::Mat cam_left_points;
    cv::Mat cam_right_points;
    int nb_point;
    nb_point = (image_points_left).size();
    if (nb_point < 3 && FIT_PLANE_CLOUD_POINT::FIT == fit_plane_cloud_point)
    {
        assert(nb_point > 3 && "To fit a plan you need a least three points");
    }

    cam_left_points = cv::Mat::zeros(1, nb_point, CV_64FC2);
    cam_right_points = cv::Mat::zeros(1, nb_point, CV_64FC2);
    from_vector_to_matrix(image_points_left, cam_left_points);
    from_vector_to_matrix(image_points_right, cam_right_points);

    cv::Mat cam_left_points_undistorsed;
    cv::Mat cam_right_points_undistorsed;
    cv::Mat cam_left_points_corrected_matches;
    cv::Mat cam_right_points_corrected_matches;

    cv::undistortPoints(cam_left_points, cam_left_points_undistorsed, M1, distCoeffs1, R1, P1);
    cv::undistortPoints(cam_right_points, cam_right_points_undistorsed, M2, distCoeffs2, R2, P2);

    cv::correctMatches(F, cam_left_points_undistorsed, cam_right_points_undistorsed, cam_left_points_corrected_matches, cam_right_points_corrected_matches);
    points3D_ = cv::Mat::zeros(4, nb_point, CV_64FC1);

    cv::Mat matPoints(nb_point, 3, CV_64FC1);
    cv::Mat matZ(nb_point, 1, CV_64FC1);
    cv::Mat matVal(nb_point, 1, CV_64FC1);
    cv::Mat res(3, 1, CV_64FC1);

    for (int idx = 0; idx < nb_point; idx++)
    {
        cv::Mat_<double> Xx(4, 1);
        cv::Point3d point_left;
        point_left.x = cam_left_points_corrected_matches.at<cv::Vec2d>(0, idx)[0];
        point_left.y = cam_left_points_corrected_matches.at<cv::Vec2d>(0, idx)[1];
        point_left.z = 1;

        cv::Point3d point_right;
        point_right.x = cam_right_points_corrected_matches.at<cv::Vec2d>(0, idx)[0];
        point_right.y = cam_right_points_corrected_matches.at<cv::Vec2d>(0, idx)[1];
        point_right.z = 1;
        Xx = IterativeLinearLSTriangulation(point_left, P1, point_right, P2);

        points3D_.at<double>(0, idx) = Xx.at<double>(0, 0);
        points3D_.at<double>(1, idx) = Xx.at<double>(1, 0);
        points3D_.at<double>(2, idx) = Xx.at<double>(2, 0);
        points3D_.at<double>(3, idx) = Xx.at<double>(3, 0);
        //cv::triangulatePoints(point_left, point_right, cam_left_points, cam_right_points, points3D_);
    }

    for (int col = 0; col < points3D_.cols; col++)
    {
        double X, Y, Z, W;
        X = points3D_.at<double>(0, col);
        Y = points3D_.at<double>(1, col);
        Z = points3D_.at<double>(2, col);
        W = points3D_.at<double>(3, col);

        cout << "X " << X << "  Y " << Y << "  Z " << Z << endl;

        if( (col+1)%81 ==0)
            cout << endl << endl << endl << endl;

        matPoints.at<double>(col, 0) = X;
        matPoints.at<double>(col, 1) = Y;
        matPoints.at<double>(col, 2) = W;
        matZ.at<double>(col, 0) = Z;
    }

    if (FIT_PLANE_CLOUD_POINT::FIT == fit_plane_cloud_point)
    {
        cv::solve(matPoints, matZ, res, cv::DECOMP_SVD );
        //cout << "res taille " << res.size() << endl;
        //// plane generation, Z*C = Ax + By + D
        pl.A = res.at<double>(0, 0);
        pl.B = res.at<double>(1, 0);
        pl.C = -1;
        pl.D = res.at<double>(2, 0);
    }/**/
}


void points_3D_reconstruction_rectified(stereo_params_cv stereo_par, std::vector<cv::Point2f> image_points_left,
                                        std::vector<cv::Point2f> image_points_right,
    cv::Mat &_3D_points, plane & _plane, FIT_PLANE_CLOUD_POINT fit_plane)
{
    plane pl;
    triangulate_plan_to_3D(stereo_par.F, stereo_par.M1, stereo_par.M2, stereo_par.P1, stereo_par.P2, stereo_par.R1, stereo_par.R2, stereo_par.T, stereo_par.D1, stereo_par.D2,
                           image_points_left,
                           image_points_right, _3D_points, fit_plane, pl);
    if (FIT_PLANE_CLOUD_POINT::FIT == fit_plane)
    {
        _plane = pl;
    }
}


void points_3D_reconstruction_rectified_(cv::Mat F, cv::Mat M1, cv::Mat M2, cv::Mat P1, cv::Mat P2, cv::Mat R1, cv::Mat R2, cv::Mat T, cv::Mat D1, cv::Mat D2, std::vector<cv::Point2f> image_points_left,
                                        std::vector<cv::Point2f> image_points_right,
    cv::Mat &_3D_points)
{
    plane pl;
    FIT_PLANE_CLOUD_POINT fit_plane = FIT_PLANE_CLOUD_POINT::FIT;

    std::vector<cv::Point2f> image_points_left_;
    std::vector<cv::Point2f> image_points_right_;

    triangulate_plan_to_3D(F, M1, M2, P1, P2, R1, R2, T, D1, D2,
                           image_points_left,
                           image_points_right, _3D_points, fit_plane, pl);
}

// This function allows to triangulate 3D points with a list of images
void points_3D_reconstruction_list_rectified(stereo_params_cv stereo_par, std::vector < std::vector<cv::Point2f>> image_points_left,
                                             std::vector<std::vector<cv::Point2f>> image_points_right, int nimages, cv::Size imageSize,
                                             std::vector<cv::Mat>& list_3D_points, std::vector<plane>& list_plane, FIT_PLANE_CLOUD_POINT fit_plane)
{
    std::cout << endl << endl << "Start triangulate points through all input images..." << endl;


    for (auto index_img = 0; index_img < nimages; index_img++)
    {
        cv::Mat points3D_;
        std::cout << "----> Triangulate points in the " << index_img + 1 << " th image " << endl;
        plane pl;
        triangulate_plan_to_3D(stereo_par.F, stereo_par.M1, stereo_par.M2, stereo_par.P1, stereo_par.P2, stereo_par.R1, stereo_par.R2, stereo_par.T, stereo_par.D1, stereo_par.D2,
                               image_points_left[index_img],
                               image_points_right[index_img], points3D_, fit_plane, pl);
        list_3D_points.push_back(points3D_);
        if (FIT_PLANE_CLOUD_POINT::FIT == fit_plane)
        {
            list_plane.push_back(pl);
        }/**/
    }
    cout << "...End triangulate points through all input images" << endl << endl;
    //system("PAUSE");
}/**/
}
