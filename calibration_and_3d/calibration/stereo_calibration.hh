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

#include "global_utils.hh"

/*#include "calibration\define_calibrate.hh"
#include "calibration\design_objet_board.hh"
#include "calibration\save_param.hh"
#include "calibration\extract_points.hh"*/

#include "mono_calibration.hh"
#include "calibration_and_3d/recontruction3D/recontruction3D.hh"

#include "algorithms/fitting/fitting_line.hh"
#include "camera_calibration.hh"

#include "algorithms/feature_extractor.hh"

 
using namespace std;

namespace tpp
{
static
bool stereo_calibration(cv::Mat cameraMatrix[2], cv::Mat distCoeffs[2],
std::vector<string> &left_images, std::vector<string> &right_images,
cv::Mat &R, cv::Mat &T, cv::Mat &E, cv::Mat &F, cv::Mat &Q,
Mode_To_Revtreive_Files mode, Type_Of_Calibration calib_type, USE_RECTIFIED_3D_POINTS_GRID rect_grid, string inputFile = "")
{
    cout << "stereo_calibration" << endl;
    std::vector<string> imageListCam1, imageListCam2;
    string inputFilename;
    if (mode == DIRECTORY)
    {
        imageListCam1 = get_list_of_files(".\\Images\\CalibrationOpticalFlow\\Camera1");
        imageListCam2 = get_list_of_files(".\\Images\\CalibrationOpticalFlow\\Camera2");
    }
    else if (mode == JSFILES)
    {
        if (inputFile.length() == 0)
        {
            inputFilename = std::string("imageList.old.json");// fichier par défaut
        }
        else
            inputFilename = inputFile;
    }

    cv::Size boardSize, imageSize;
    float squareSize;

    //Mat cameraMatrix[2], distCoeffs[2];
    string outputFilename;

    int flags = 0;

    std::vector<std::vector<cv::Point2f> > imagePoints[2];

    std::vector<std::vector<cv::Point2f> > imagePoints_C[2];

    std::vector<string> goodImageList1;
    std::vector<string> goodImageList2;
    Pattern pattern;

    boardSize.width = 6;
    boardSize.height = 6;
    pattern = CIRCLES_GRID;
    squareSize = 160;
    outputFilename = std::string("out_stereo_camera_data.json");
    flags |=  cv::CALIB_FIX_ASPECT_RATIO ;
    flags |=  cv::CALIB_FIX_PRINCIPAL_POINT;
    flags |= cv::CALIB_USE_INTRINSIC_GUESS ;
    flags |= cv::CALIB_ZERO_TANGENT_DIST;

    int i, j, k, nimages;

    // extrait les noms des images a partir du fichier json
    //remplirListes(inputFilename, imageListCam1, imageListCam2);


    nimages = imageListCam1.size();

    assert(nimages >= 2 && "The calibration process needs at least 2 images pairs");

    assert(imageListCam1.size() == imageListCam2.size() && "The number of images are not the same");

    cout << "The number of images is " << nimages << endl;

    bool extrac_ok;
    int number_points = boardSize.height*boardSize.width;
    int nb_ok = 0;

    imagePoints[0].clear();
    imagePoints[1].clear();
    imagePoints[0].resize(nimages);
    imagePoints[1].resize(nimages);

    cv::FileStorage fs_pt("points.yml", cv::FileStorage::WRITE);

    ofstream image1X;
    image1X.open("points1X.txt");
    ofstream image1Y;
    image1Y.open("points1Y.txt");

    ofstream image2X;
    image2X.open("pointsCX.txt");
    ofstream image2Y;
    image2Y.open("pointsCY.txt");

    int idx_ok = 0;

    for (int i = 0; i < (int)imageListCam1.size(); i++)
    {

        cout << "boucle" << endl;
        cv::Mat view = cv::imread(imageListCam1[i], CV_16U);
        std::vector<cv::Point2f> im_point1;

        extrac_ok = extrairePointsImage(view, boardSize, im_point1, imageListCam1[i]);
        //assert(1 == 2 && "1!=2");
        if (!extrac_ok)
        {
            cout << "not ok pair 1" << endl;
            continue;
        }
        cout << "Image pair number : " << i << endl << endl;
        //cout << endl << endl << endl;

        cv::Mat view1 = cv::imread(imageListCam2[i], CV_16U);
        std::vector<cv::Point2f> im_point2;
        extrac_ok = extrairePointsImage(view1, boardSize, im_point2, imageListCam2[i]);

        if (!extrac_ok)
        {
            cout << "not ok pair 2 " << endl;
            continue;
        }

        imagePoints[0][idx_ok] = im_point1;
        imagePoints[1][idx_ok] = im_point2;
        goodImageList1.push_back(imageListCam1[i]);
        goodImageList2.push_back(imageListCam2[i]);
        nb_ok++;
        idx_ok++;
        imageSize = view.size();
    }

    for (int i = 0; i < nb_ok; i++)
    {
        cv::Mat img1, img2;
        img1 = cv::Mat::zeros(imageSize, CV_8U);
        img2 = cv::Mat::zeros(imageSize, CV_8U);
        std::vector<cv::Point2f> im_pt1 = imagePoints[0][i];
        std::vector<cv::Point2f> im_pt2 = imagePoints[1][i];
        int nb_points = im_pt1.size();
        for (int k = 0; k < nb_points; k++)
        {
            int x, y;
            img1.at<uchar>(im_pt1[k]) = 255;
            img2.at<uchar>(im_pt2[k]) = 255;
        }
        string file_name;
        file_name = std::string("./Images/imageleft").append(std::to_string(i)).append(".bmp");
        left_images.push_back(file_name);
        cv::imwrite(file_name, img1);
        file_name = std::string("./Images/imageright").append(std::to_string(i)).append(".bmp");
        cv::imwrite(file_name, img2);
        right_images.push_back(file_name);
    }

    nimages = nb_ok;

    assert(nimages >= 2 && "The calibration process needs at least 2 images pairs");

    cout << "taille " << (imagePoints[0]).size() << "   " << (imagePoints[1]).size() << " nimages " << nimages << endl;

    //assert((imagePoints[0]).size() == (imagePoints[1]).size() && (imagePoints[0]).size() == nimages && "The extraction is not well performed");

    std::vector<std::vector<cv::Point3f>> objectPoints;

    (imagePoints[0]).resize(nimages);
    (imagePoints[1]).resize(nimages);
    objectPoints.resize(nimages);

    //system("PAUSE");

    cout << "taille " << imagePoints[0].size() << endl;

    CalculateObjectsPoints_Stereo(boardSize, squareSize, objectPoints, nimages, pattern);

    cout << "initialization " << endl;


    cameraMatrix[0] = initCameraMatrix2D(objectPoints, imagePoints[0], imageSize, 0);
    cameraMatrix[1] = initCameraMatrix2D(objectPoints, imagePoints[1], imageSize, 0);

    cout << "end of initialization " << endl;

    cout << "start calibration " << endl;

    double err = stereoCalibrate(objectPoints, (imagePoints[0]), (imagePoints[1]),
            cameraMatrix[0], distCoeffs[0],
            cameraMatrix[1], distCoeffs[1], imageSize, R, T, E, F,
        cv::CALIB_FIX_ASPECT_RATIO +
        cv::CALIB_ZERO_TANGENT_DIST +
        cv::CALIB_USE_INTRINSIC_GUESS +
        cv::CALIB_SAME_FOCAL_LENGTH +
        cv::CALIB_RATIONAL_MODEL +
        cv::CALIB_FIX_K3 + cv::CALIB_FIX_K4 + cv::CALIB_FIX_K5,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 500, 1e-15)
            );
    //attention aux flags!
    //enregistrement des parametres
    saveStereoParams(outputFilename,
                     imageSize, cameraMatrix[0], cameraMatrix[1],
            distCoeffs[0], distCoeffs[1],
            R, T, E, F, err);

    cout << "end of callibration" << endl;

    //system("PAUSE");

    // CALIBRATION QUALITY CHECK
    // because the output fundamental matrix implicitly
    // includes all the output information,
    // we can check the quality of calibration using the
    // epipolar geometry constraint: m2^t*F*m1=0

    err = 0;
    int npoints = 0;
    std::vector<cv::Vec3f> lines[2];
    for (i = 0; i < nimages; i++)
    {
        int npt = (int)imagePoints[0][i].size();
        cv::Mat imgpt[2];
        for (k = 0; k < 2; k++)
        {
            imgpt[k] = cv::Mat(imagePoints[k][i]);
            cv::undistortPoints(imgpt[k], imgpt[k], cameraMatrix[k], distCoeffs[k], cv::Mat(), cameraMatrix[k]);
            cv::computeCorrespondEpilines(imgpt[k], k + 1, F, lines[k]);
        }
        for (j = 0; j < npt; j++)
        {
            double errij = fabs(imagePoints[0][i][j].x*lines[1][j][0] +
                    imagePoints[0][i][j].y*lines[1][j][1] + lines[1][j][2]) +
                    fabs(imagePoints[1][i][j].x*lines[0][j][0] +
                    imagePoints[1][i][j].y*lines[0][j][1] + lines[0][j][2]);
            err += errij;
        }
        npoints += npt;
    }
    cout << "average epipolar err = " << err / npoints << endl;

    // save intrinsic parameters
    cv::FileStorage fs("intrinsics.yml", cv::FileStorage::WRITE);
    if (fs.isOpened())
    {
        fs << "M1" << cameraMatrix[0] << "D1" << distCoeffs[0] <<
              "M2" << cameraMatrix[1] << "D2" << distCoeffs[1];
        fs.release();
    }
    else
        cout << "Error: can not save the intrinsic parameters\n";

    cv::Mat R1, R2, P1, P2;
    cv::Rect validRoi[2];

    //stereoRectify(cameraMatrix[0], distCoeffs[0],
    //cameraMatrix[1], distCoeffs[1],
    //	imageSize, R, T, R1, R2, P1, P2, Q);

    cv::stereoRectify(cameraMatrix[0], distCoeffs[0],
            cameraMatrix[1], distCoeffs[1],
            imageSize, R, T, R1, R2, P1, P2, Q,
        cv::CALIB_ZERO_DISPARITY, 1, imageSize, &validRoi[0], &validRoi[1]);

    if (USE_RECTIFIED_3D_POINTS_GRID::YES == rect_grid)
    {
        //points_3D_reconstruction_rectified(F, cameraMatrix[0], cameraMatrix[1], P1, P2, R1, R2, T, distCoeffs[0], distCoeffs[1],
        //	imagePoints[0], imagePoints[1], nimages, imageSize);
    }

    fs.open("extrinsics.yml", cv::FileStorage::WRITE);
    if (fs.isOpened())
    {
        fs << "R" << R << "T" << T << "T norme" << cv::norm(T) << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q << "F" << F;
        fs.release();
    }
    else
        cout << "Error: can not save the extrinsic parameters\n";

    return true;
}/**/



struct rangegenerator {
    rangegenerator(int init) : start(init) { }

    int operator()() {
        return start++;
    }

    int start;
};








static
bool stereo_calibration_routine(cv::Mat cameraMatrix[2], cv::Mat distCoeffs[2],
std::vector<string> &left_images, std::vector<string> &right_images,
cv::Mat &R, cv::Mat &T, cv::Mat &E, cv::Mat &F, cv::Mat &Q,
Mode_To_Revtreive_Files mode, Type_Of_Calibration calib_type, USE_RECTIFIED_3D_POINTS_GRID rect_grid, string inputFile = "")
{
    cout << "stereo_calibration_routine " << endl;

    std::vector<string> imageListCam1, imageListCam2;
    typedef std::vector<cv::Point2f> prise;
    typedef std::vector<std::vector<prise>> liste_prise_stereo;
    string inputFilename;

    if (mode == DIRECTORY)
    {
        //getListFilesOfDirectory(imageListCam1, imageListCam2);
    }
    else if (mode == JSFILES)
    {
        if (inputFile.length() == 0)
        {
            inputFilename = std::string("./json/grid_4.json");// fichier par défaut
        }
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

    int flags = 0;

    liste_prise_stereo imagePoints_C;
    std::vector<std::vector<cv::Point2f> > imagePoints[2];

    std::vector<std::string> goodImageList1;
    std::vector<std::string> goodImageList2;
    Pattern pattern;

    boardSize.width = 4;
    boardSize.height = 4;
    pattern = CIRCLES_GRID;
    squareSize = 160;
    outputFilename = std::string("out_stereo_camera_data.json");
    flags |=  cv::CALIB_FIX_ASPECT_RATIO ;
    flags |=  cv::CALIB_FIX_PRINCIPAL_POINT;
    flags |= cv::CALIB_USE_INTRINSIC_GUESS ;
    flags |= cv::CALIB_ZERO_TANGENT_DIST;

    int i, j, k, nimages;

    // extrait les noms des images a partir du fichier json
    remplirListes(inputFilename, imageListCam1, imageListCam2);
    nimages = imageListCam1.size();

    assert(nimages >= 2 && "The calibration process needs at least 2 images pairs");

    assert(imageListCam1.size() == imageListCam2.size() && "The number of images are not the same");

    bool extrac_ok;
    int number_points = boardSize.height*boardSize.width;
    int nb_ok = 0;

    imagePoints_C.clear();
    imagePoints_C.resize(nimages);

    for (int i = 0; i < nimages; i++)
    {
        imagePoints_C[i].resize(2);
    }

    cv::FileStorage fs_pt("points.yml", cv::FileStorage::WRITE);

    ofstream image1X;
    image1X.open("points1X.txt");
    ofstream image1Y;
    image1Y.open("points1Y.txt");

    ofstream image2X;
    image2X.open("pointsCX.txt");
    ofstream image2Y;
    image2Y.open("pointsCY.txt");

    int idx_ok = 0;

    for (int i = 0; i < (int)imageListCam1.size(); i++)
    {

        //cout << imageListCam1[i] << "lkjlkjjkglj" << endl;
        cv::Mat view = cv::imread(imageListCam1[i], CV_16U);
        std::vector<cv::Point2f> im_point1;
        //cout << "extrairePointsImage " << i << endl;
        cv::Mat img_harris = view.clone();
        //harris_detector_settings hst;
        //default_initialization_harris_settings(hst);
        //std::vector<Point2f> corners_harris;
        //extract_corners_harris_(img_harris, corners_harris, hst, i, 0);
        extrac_ok = extrairePointsImage(view, boardSize, im_point1, imageListCam1[i]);
        //assert(1 == 2 && "1!=2");
        if (!extrac_ok)
        {
            cout << "not ok pair 1" << endl;
            continue;
        }
        cout << "Image pair number : " << i << endl << endl;
        //cout << endl << endl << endl;

        cv::Mat view1 = cv::imread(imageListCam2[i], CV_16U);
        std::vector<cv::Point2f> im_point2;
        extrac_ok = extrairePointsImage(view1, boardSize, im_point2, imageListCam2[i]);

        if (!extrac_ok)
        {
            cout << "not ok pair 2 " << endl;
            continue;
        }

        imagePoints_C[idx_ok][0] = im_point1;
        imagePoints_C[idx_ok][1] = im_point2;
        goodImageList1.push_back(imageListCam1[i]);
        goodImageList2.push_back(imageListCam2[i]);
        nb_ok++;
        idx_ok++;
        imageSize = view.size();
    }

    for (int i = 0; i < nb_ok; i++)
    {
        cv::Mat img1, img2;
        img1 = cv::Mat::zeros(imageSize, CV_8U);
        img2 = cv::Mat::zeros(imageSize, CV_8U);
        std::vector<cv::Point2f> im_pt1 = imagePoints_C[i][0];
        std::vector<cv::Point2f> im_pt2 = imagePoints_C[i][1];
        int nb_points = im_pt1.size();
        for (int k = 0; k < nb_points; k++)
        {
            int x, y;
            img1.at<uchar>(im_pt1[k]) = 255;
            img2.at<uchar>(im_pt2[k]) = 255;
        }
        string file_name;
        file_name = std::string("./Images/imageleft").append(std::to_string(i)).append(".bmp");
        left_images.push_back(file_name);
        //imwrite(file_name, img1);
        file_name = std::string("./Images/imageright").append(std::to_string(i)).append(".bmp");
        //imwrite(file_name, img2);
        right_images.push_back(file_name);
    }

    nimages = nb_ok;

    assert(nimages >= 2 && "The calibration process needs at least 2 images pairs");

    (imagePoints_C).resize(nimages);

    //cout << "taille " << (imagePoints[0]).size() << "   " << (imagePoints[1]).size() << " nimages " << nimages << endl;

    //assert((imagePoints[0]).size() == (imagePoints[1]).size() && (imagePoints[0]).size() == nimages && "The extraction is not well performed");

    //system("PAUSE");

    //cout << "taille " << imagePoints_C[0][0].size() << endl;

    //cout << "objectPoints " << objectPoints.size() << endl;


    cout << "nimages " << nimages << endl;
    //cout << "ni " << 10 << endl;

    for (int cp = 40; cp <= nimages-10; cp = cp + 5)
    {
        int nb_iter = 100;
        int iter = 0;

        for (iter = 0; iter < nb_iter; iter++)
        {
            std::vector<int> indices(nimages, 0);

            std::vector<int> taken(cp);

            generate(begin(indices), end(indices), rangegenerator(0));

            std::random_device rd;
            std::mt19937 g(rd());

            //random_shuffle(begin(indices), end(indices));

            std::shuffle(indices.begin(), indices.end(), g);

            std::vector<std::vector<cv::Point3f>> objectPoints;

            objectPoints.resize(cp);

            imagePoints[0].clear();
            imagePoints[1].clear();
            imagePoints[0].resize(cp);
            imagePoints[1].resize(cp);

            cout << "first loop cp = " << cp << "   " << iter << " " << indices.size() << endl;

            for (int i = 0; i < cp; i++)
            {
                //cout << indices[i] << endl;
                taken[i] = indices[i];
                imagePoints[0][i] = imagePoints_C[indices[i]][0];
                imagePoints[1][i] = imagePoints_C[indices[i]][1];
            }

            CalculateObjectsPoints_Stereo(boardSize, squareSize, objectPoints, cp, pattern);

            cout << "initialization " << endl;

            cameraMatrix[0] = initCameraMatrix2D(objectPoints, imagePoints[0], imageSize, 0);
            cameraMatrix[1] = initCameraMatrix2D(objectPoints, imagePoints[1], imageSize, 0);

            cout << "end of initialization " << endl;

            cout << "start calibration " << endl;


            double err = stereoCalibrate(objectPoints, (imagePoints[0]), (imagePoints[1]),
                    cameraMatrix[0], distCoeffs[0],
                    cameraMatrix[1], distCoeffs[1], imageSize, R, T, E, F,
                cv::CALIB_FIX_ASPECT_RATIO +
                cv::CALIB_ZERO_TANGENT_DIST +
                cv::CALIB_SAME_FOCAL_LENGTH +
                cv::CALIB_RATIONAL_MODEL +
                cv::CALIB_FIX_K3 + cv::CALIB_FIX_K4 + cv::CALIB_FIX_K5,
                cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 500, 1e-9)
                    );

            double rms = err;

            cout << "average epipolar err = " << err  << endl;

            //attention aux flags!
            //enregistrement des parametres

            cout << "end of callibration" << endl;

            //system("PAUSE");

            // CALIBRATION QUALITY CHECK
            // because the output fundamental matrix implicitly
            // includes all the output information,
            // we can check the quality of calibration using the
            // epipolar geometry constraint: m2^t*F*m1=0

            err = 0;
            int npoints = 0;
            std::vector<cv::Vec3f> lines[2];
            for (i = 0; i < cp; i++)
            {
                int npt = (int)imagePoints[0][i].size();
                cv::Mat imgpt[2];
                for (k = 0; k < 2; k++)
                {
                    imgpt[k] = cv::Mat(imagePoints[k][i]);
                    cv::undistortPoints(imgpt[k], imgpt[k], cameraMatrix[k], distCoeffs[k], cv::Mat(), cameraMatrix[k]);
                    cv::computeCorrespondEpilines(imgpt[k], k + 1, F, lines[k]);
                }
                for (j = 0; j < npt; j++)
                {
                    double errij = fabs(imagePoints[0][i][j].x*lines[1][j][0] +
                            imagePoints[0][i][j].y*lines[1][j][1] + lines[1][j][2]) +
                            fabs(imagePoints[1][i][j].x*lines[0][j][0] +
                            imagePoints[1][i][j].y*lines[0][j][1] + lines[0][j][2]);
                    err += errij;
                }
                npoints += npt;
            }

            float e = err / npoints;


            cout << "e = " << err / npoints << "  points " << npoints << endl;



            if (rms > 1 && rms < 0.1)
            {
                cout << "tooooo big" << endl << endl << endl;
                continue;
            }


            std::string intrinsic = std::string("./yml4/intrinsics").append(std::to_string(cp)).append(std::string("_")).append(std::to_string(iter)).append(std::string(".yml"));

            // save intrinsic parameters
            cv::FileStorage fs(intrinsic, cv::FileStorage::WRITE);
            if (fs.isOpened())
            {
                fs << "M1" << cameraMatrix[0] << "D1" << distCoeffs[0] <<
                      "M2" << cameraMatrix[1] << "D2" << distCoeffs[1];
                fs.release();
            }
            else
                cout << "Error: can not save the intrinsic parameters\n";

            cv::Mat R1, R2, P1, P2;
            cv::Rect validRoi[2];

            //stereoRectify(cameraMatrix[0], distCoeffs[0],
            //cameraMatrix[1], distCoeffs[1],
            //imageSize, R, T, R1, R2, P1, P2, Q);

            cv::stereoRectify(cameraMatrix[0], distCoeffs[0],
                    cameraMatrix[1], distCoeffs[1],
                    imageSize, R, T, R1, R2, P1, P2, Q,
                cv::CALIB_ZERO_DISPARITY, 1, imageSize, &validRoi[0], &validRoi[1]);



            if (USE_RECTIFIED_3D_POINTS_GRID::YES == rect_grid)
            {
                stereo_params_cv stereo_par;
                stereo_par.retreive_values();
                cv::Mat _3D_points;

                plane pl;

                cv::Point2f l = imagePoints[0][0][10];
                cv::Point2f r = imagePoints[1][0][10];
                std::vector<cv::Point2f> object_points_left;
                std::vector<cv::Point2f> object_points_right;

                cout << "left " << l << " right " << r << endl;

                augment_data_around_point(l,r,3,object_points_left,object_points_right);

                //points_3D_reconstruction_rectified_(F, cameraMatrix[0], cameraMatrix[1], P1, P2, R1, R2, T, distCoeffs[0], distCoeffs[1],
                //    imagePoints[0][0], imagePoints[1][0],_3D_points);

                points_3D_reconstruction_rectified(stereo_par, object_points_left, object_points_right, _3D_points, pl, FIT_PLANE_CLOUD_POINT::FIT);

                save_3D_points(_3D_points);

                /*void points_3D_reconstruction_rectified(Mat F,Mat M1,Mat M2,Mat P1,Mat P2,Mat R1,Mat R2,Mat T,Mat D1, Mat D2, std::vector<Point2f> image_points_left,
                                        std::vector<Point2f> image_points_right,
                                        Mat &_3D_points)*/
            }

            std::string extrinsic = std::string("./yml4/extrinsics").append(std::to_string(cp)).append(std::string("_")).append(std::to_string(iter)).append(std::string(".yml"));

            fs.open(extrinsic, cv::FileStorage::WRITE);
            if (fs.isOpened())
            {
                fs << "R" << R << "T" << T << "T norme" << cv::norm(T) << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q << "F" << F << "E" << E
                   << "list_images" << taken << "average_epipolar_error" << err / npoints << "rms" << rms;
                fs.release();
            }
            else
                cout << "Error: can not save the extrinsic parameters\n";

            cout << endl << endl << endl;

            // OpenCV can handle left-right
            // or up-down camera arrangements
            bool isVerticalStereo = false;

            // COMPUTE AND DISPLAY RECTIFICATION
            if (!showRectified)
                return true;

            cv::Mat rmap[2][2];
            // IF BY CALIBRATED (BOUGUET'S METHOD)
            if (useCalibrated)
            {
                // we already computed everything
            }
            // OR ELSE HARTLEY'S METHOD
            else
                // use intrinsic parameters of each camera, but
                // compute the rectification transformation directly
                // from the fundamental matrix
            {
                std::vector<cv::Point2f> allimgpt[2];
                for (k = 0; k < 2; k++)
                {
                    for (i = 0; i < cp; i++)
                        std::copy(imagePoints[k][i].begin(), imagePoints[k][i].end(), std::back_inserter(allimgpt[k]));
                }
                F = cv::findFundamentalMat(cv::Mat(allimgpt[0]), cv::Mat(allimgpt[1]), cv::FM_8POINT, 0, 0);
                cv::Mat H1, H2;
                stereoRectifyUncalibrated(cv::Mat(allimgpt[0]), cv::Mat(allimgpt[1]), F, imageSize, H1, H2, 3);

                R1 = cameraMatrix[0].inv()*H1*cameraMatrix[0];
                R2 = cameraMatrix[1].inv()*H2*cameraMatrix[1];
                P1 = cameraMatrix[0];
                P2 = cameraMatrix[1];
            }

            //Precompute maps for cv::remap()
            cv::initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
            cv::initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

            cv::Mat canvas;
            double sf;
            int w, h;
            if (!isVerticalStereo)
            {
                sf = 600. / MAX(imageSize.width, imageSize.height);
                w = cvRound(imageSize.width*sf);
                h = cvRound(imageSize.height*sf);
                canvas.create(h, w * 2, CV_8UC3);
            }
            else
            {
                sf = 300. / MAX(imageSize.width, imageSize.height);
                w = cvRound(imageSize.width*sf);
                h = cvRound(imageSize.height*sf);
                canvas.create(h * 2, w, CV_8UC3);
            }

            for (i = 0; i < cp; i++)
            {
                for (k = 0; k < 2; k++)
                {
                    cv::Mat img, rimg, cimg;
                    if (k == 0)
                    {
                        img = cv::imread(left_images[i], 0);
                    }
                    else
                    {
                        img = cv::imread(right_images[i], 0);
                    }
                    cv::remap(img, rimg, rmap[k][0], rmap[k][1], cv::INTER_LANCZOS4);
                    string file_na;
                    file_na = std::string("./Images/recti_").append(std::to_string(cp)).append(std::string("_")).append(std::to_string(k))
                            .append(std::string("_")).append(std::to_string(i)).append(".bmp");
                    cv::cvtColor(rimg, cimg, cv::COLOR_GRAY2BGR);
                    //imwrite(file_na, rimg);
                    file_na = std::string("./Images/recti__").append(std::to_string(cp)).append(std::string("_")).append(std::to_string(k))
                            .append(std::string("_")).append(std::to_string(i)).append(".bmp");
                    //imwrite(file_na, img);
                    cv::Mat canvasPart = !isVerticalStereo ? canvas(cv::Rect(w*k, 0, w, h)) : canvas(cv::Rect(0, h*k, w, h));
                    cv::resize(cimg, canvasPart, canvasPart.size(), 0, 0, cv::INTER_AREA);
                    if (useCalibrated)
                    {
                        cv::Rect vroi(cvRound(validRoi[k].x*sf), cvRound(validRoi[k].y*sf),
                                  cvRound(validRoi[k].width*sf), cvRound(validRoi[k].height*sf));
                        cv::rectangle(canvasPart, vroi, cv::Scalar(0, 0, 255), 3, 8);
                    }
                }

                if (!isVerticalStereo)
                    for (j = 0; j < canvas.rows; j += 16)
                        line(canvas, cv::Point(0, j), cv::Point(canvas.cols, j), cv::Scalar(0, 255, 0), 1, 8);
                else
                    for (j = 0; j < canvas.cols; j += 16)
                        line(canvas, cv::Point(j, 0), cv::Point(j, canvas.rows), cv::Scalar(0, 255, 0), 1, 8);
                string file_name;
                file_name = std::string("./Images/rectified_").append(std::to_string(cp)).append(std::string("_")).append(std::to_string(i)).append(".bmp");
                //imwrite(file_name, canvas);
                //imshow("rectified", canvas);
                //char c = (char)waitKey();
                //if (c == 27 || c == 'q' || c == 'Q')
                //	break;
            }


        }

        //return true;


    }

    return true;
}/**/



static
bool stereo_calibration_liste(cv::Mat cameraMatrix[2], cv::Mat distCoeffs[2],
std::vector<string> &left_images, std::vector<string> &right_images,
cv::Mat &R, cv::Mat &T, cv::Mat &E, cv::Mat &F, cv::Mat &Q,
Mode_To_Revtreive_Files mode, Type_Of_Calibration calib_type, USE_RECTIFIED_3D_POINTS_GRID rect_grid,VectorXf list_img, string inputFile = "")
{
    cout << "stereo_calibration_routine " << endl;

    std::vector<string> imageListCam1, imageListCam2;
    typedef std::vector<cv::Point2f> prise;
    typedef std::vector<std::vector<prise>> liste_prise_stereo;
    string inputFilename;

    if (mode == DIRECTORY)
    {
        //getListFilesOfDirectory(imageListCam1, imageListCam2);
    }
    else if (mode == JSFILES)
    {
        if (inputFile.length() == 0)
        {
            inputFilename = std::string("./json/images.json");// fichier par défaut
        }
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

    int flags = 0;

    liste_prise_stereo imagePoints_C;
    std::vector<std::vector<cv::Point2f> > imagePoints[2];

    std::vector<string> goodImageList1;
    std::vector<string> goodImageList2;
    Pattern pattern;

    boardSize.width = 6;
    boardSize.height = 6;
    pattern = CIRCLES_GRID;
    squareSize = 160;
    outputFilename = std::string("out_stereo_camera_data.json");
    flags |=  cv::CALIB_FIX_ASPECT_RATIO ;
    flags |=  cv::CALIB_FIX_PRINCIPAL_POINT;
    flags |= cv::CALIB_USE_INTRINSIC_GUESS ;
    flags |= cv::CALIB_ZERO_TANGENT_DIST;
    int i, j, k, nimages;

    // extrait les noms des images a partir du fichier json
    remplirListes(inputFilename, imageListCam1, imageListCam2);
    nimages = imageListCam1.size();

    assert(nimages >= 2 && "The calibration process needs at least 2 images pairs");

    assert(imageListCam1.size() == imageListCam2.size() && "The number of images are not the same");

    bool extrac_ok;
    int number_points = boardSize.height*boardSize.width;
    int nb_ok = 0;

    imagePoints_C.clear();
    imagePoints_C.resize(nimages);

    for (int i = 0; i < nimages; i++)
    {
        imagePoints_C[i].resize(2);
    }

    cv::FileStorage fs_pt("points.yml", cv::FileStorage::WRITE);

    ofstream image1X;
    image1X.open("points1X.txt");
    ofstream image1Y;
    image1Y.open("points1Y.txt");

    ofstream image2X;
    image2X.open("pointsCX.txt");
    ofstream image2Y;
    image2Y.open("pointsCY.txt");

    int idx_ok = 0;

    for (int i = 0; i < (int)imageListCam1.size(); i++)
    {
        cv::Mat view = cv::imread(imageListCam1[i], CV_16U);
        std::vector<cv::Point2f> im_point1;
        //cout << "extrairePointsImage " << i << endl;
        cv::Mat img_harris = view.clone();
        //harris_detector_settings hst;
        //default_initialization_harris_settings(hst);
        //std::vector<Point2f> corners_harris;
        //extract_corners_harris_(img_harris, corners_harris, hst, i, 0);
        extrac_ok = extrairePointsImage(view, boardSize, im_point1, imageListCam1[i]);
        //assert(1 == 2 && "1!=2");
        if (!extrac_ok)
        {
            cout << "not ok pair 1" << endl;
            continue;
        }
        cout << "Image pair number : " << i << endl << endl;
        //cout << endl << endl << endl;

        cv::Mat view1 = cv::imread(imageListCam2[i], CV_16U);
        std::vector<cv::Point2f> im_point2;
        extrac_ok = extrairePointsImage(view1, boardSize, im_point2, imageListCam2[i]);

        if (!extrac_ok)
        {
            cout << "not ok pair 2 " << endl;
            continue;
        }

        imagePoints_C[idx_ok][0] = im_point1;
        imagePoints_C[idx_ok][1] = im_point2;
        goodImageList1.push_back(imageListCam1[i]);
        goodImageList2.push_back(imageListCam2[i]);
        nb_ok++;
        idx_ok++;
        imageSize = view.size();
    }



    nimages = nb_ok;

    assert(nimages >= 2 && "The calibration process needs at least 2 images pairs");

    (imagePoints_C).resize(nimages);





    int cp = 35;
    std::vector<int> indices(nimages, 0);

    std::vector<int> taken(cp);



    //random_shuffle(begin(indices), end(indices));

    std::vector<std::vector<cv::Point3f>> objectPoints;

    objectPoints.resize(cp);

    imagePoints[0].clear();
    imagePoints[1].clear();
    imagePoints[0].resize(cp);
    imagePoints[1].resize(cp);

    for (int i = 0; i < cp; i++)
    {
        //cout << indices[i] << endl;
        taken[i] = list_img[i];
        imagePoints[0][i] = imagePoints_C[list_img[i]][0];
        imagePoints[1][i] = imagePoints_C[list_img[i]][1];
    }

    CalculateObjectsPoints_Stereo(boardSize, squareSize, objectPoints, cp, pattern);

    cout << "initialization " << endl;

    cameraMatrix[0] = initCameraMatrix2D(objectPoints, imagePoints[0], imageSize, 0);
    cameraMatrix[1] = initCameraMatrix2D(objectPoints, imagePoints[1], imageSize, 0);

    cout << "end of initialization " << endl;

    cout << "start calibration " << endl;


    double err = stereoCalibrate(objectPoints, (imagePoints[0]), (imagePoints[1]),
            cameraMatrix[0], distCoeffs[0],
            cameraMatrix[1], distCoeffs[1], imageSize, R, T, E, F,
        cv::CALIB_FIX_ASPECT_RATIO +
        cv::CALIB_ZERO_TANGENT_DIST +
        cv::CALIB_SAME_FOCAL_LENGTH  +
        cv::CALIB_RATIONAL_MODEL +
        cv::CALIB_FIX_K3 + cv::CALIB_FIX_K4 + cv::CALIB_FIX_K5/**/,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 1e-6)
            );

    double rms = err;

    cout << "average epipolar err = " << err  << endl;

    //attention aux flags!
    //enregistrement des parametres

    cout << "end of callibration" << endl;

    //system("PAUSE");

    // CALIBRATION QUALITY CHECK
    // because the output fundamental matrix implicitly
    // includes all the output information,
    // we can check the quality of calibration using the
    // epipolar geometry constraint: m2^t*F*m1=0

    err = 0;
    int npoints = 0;
    std::vector<cv::Vec3f> lines[2];
    std::vector<cv::Mat> list_mat;
    std::vector<cv::Mat> list_mat_;
    for (i = 0; i < cp; i++)
    {
        int npt = (int)imagePoints[0][i].size();
        cv::Mat imgpt[2];
        for (k = 0; k < 2; k++)
        {
            imgpt[k] = cv::Mat(imagePoints[k][i]);

            cv::undistortPoints(imgpt[k], imgpt[k], cameraMatrix[k], distCoeffs[k], cv::Mat(), cameraMatrix[k]);
            cv::computeCorrespondEpilines(imgpt[k], k + 1, F, lines[k]);
        }
        //cout << "lines " << lines[0][0] << endl;
        for (j = 0; j < npt; j++)
        {
            double errij = fabs(imagePoints[0][i][j].x*lines[1][j][0] +
                    imagePoints[0][i][j].y*lines[1][j][1] + lines[1][j][2]) +
                    fabs(imagePoints[1][i][j].x*lines[0][j][0] +
                    imagePoints[1][i][j].y*lines[0][j][1] + lines[0][j][2]);
            err += errij;
        }
        list_mat.push_back(imgpt[0]);
        list_mat_.push_back(imgpt[1]);
        npoints += npt;
    }

    for (int i = 0; i < cp; i++)
    {
        cv::Mat img1, img2;
        img1 = cv::Mat::zeros(imageSize, CV_8UC3);
        img2 = cv::Mat::zeros(imageSize, CV_8UC3);
        //vector<Point2f> im_pt1 = imagePoints_C[i][0];
        std::vector<cv::Point2f> im_pt1 = list_mat[i];
        std::vector<cv::Point2f> im_pt2 = list_mat_[i];
        int nb_points = im_pt1.size();
        for (int k = 0; k < nb_points; k++)
        {
            int x, y;
            //cout << im_pt1[k] << endl;
            img1.at<cv::Vec3b>(im_pt1[k]) = cv::Vec3b(255,255,255);
            img2.at<cv::Vec3b>(im_pt2[k]) = cv::Vec3b(255,255,255);
        }
        for(int l = 0 ; l < lines[0].size() ; l++)
        {
            float a,b,c;
            a = lines[0][l][0];
            b = lines[0][l][1];
            c = lines[0][l][2];
            cv::Point p1,p2;
            p1.x = 0;
            p1.y = int(-c/b);
            p2.x = 80;
            p2.y = int(-c/b - (80.0*a)/b);
            cout << "pt1 " << p1 << " pt2 " << p2 << endl;
            line(img2, p1, p2, cv::Scalar(0, 255, 0), 1, 1);
            for(int w = 0 ; w < 80 ; w++)
            {
                int r = int( (- c - (a*w))/ b);
                cout << "x " << w << "  y " << r << endl;
            }
        }
        string file_name;
        file_name = std::string("./ymlo/imageleft").append(std::to_string(i)).append(".bmp");
        left_images.push_back(file_name);
        imwrite(file_name, img1);
        file_name = std::string("./ymlo/imageright").append(std::to_string(i)).append(".bmp");
        imwrite(file_name, img2);
        right_images.push_back(file_name);
    }

    float e = err / npoints;


    cout << "e = " << err / npoints << "  points " << npoints << endl;


    std::string intrinsic = std::string("./ymlo/intrinsics").append(std::to_string(cp)).append(std::string(".yml"));

    // save intrinsic parameters
    cv::FileStorage fs(intrinsic, cv::FileStorage::WRITE);
    if (fs.isOpened())
    {
        fs << "M1" << cameraMatrix[0] << "D1" << distCoeffs[0] <<
              "M2" << cameraMatrix[1] << "D2" << distCoeffs[1];
        fs.release();
    }
    else
        cout << "Error: can not save the intrinsic parameters\n";

    cv::Mat R1, R2, P1, P2;
    cv::Rect validRoi[2];

    //stereoRectify(cameraMatrix[0], distCoeffs[0],
    //cameraMatrix[1], distCoeffs[1],
    //imageSize, R, T, R1, R2, P1, P2, Q);

    cv::stereoRectify(cameraMatrix[0], distCoeffs[0],
            cameraMatrix[1], distCoeffs[1],
            imageSize, R, T, R1, R2, P1, P2, Q,
        cv::CALIB_ZERO_DISPARITY, 1, imageSize, &validRoi[0], &validRoi[1]);

    if (USE_RECTIFIED_3D_POINTS_GRID::YES == rect_grid)
    {
        //points_3D_reconstruction_rectified(F, cameraMatrix[0], cameraMatrix[1], P1, P2, R1, R2, T, distCoeffs[0], distCoeffs[1],
        //	imagePoints[0], imagePoints[1], nimages, imageSize);
    }

    std::string extrinsic = std::string("./ymlo/extrinsics").append(std::to_string(cp)).append(std::string(".yml"));

    fs.open(extrinsic, cv::FileStorage::WRITE);
    if (fs.isOpened())
    {
        fs << "R" << R << "T" << T << "T norme" << cv::norm(T) << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q << "F" << F << "E" << E
           << "list_images" << taken << "average_epipolar_error" << err / npoints << "rms" << rms;
        fs.release();
    }
    else
        cout << "Error: can not save the extrinsic parameters\n";

    cout << endl << endl << endl;

    // OpenCV can handle left-right
    // or up-down camera arrangements
    bool isVerticalStereo = false;

    // COMPUTE AND DISPLAY RECTIFICATION
    if (!showRectified)
        return true;

    cv::Mat rmap[2][2];
    // IF BY CALIBRATED (BOUGUET'S METHOD)
    if (useCalibrated)
    {
        // we already computed everything
    }
    // OR ELSE HARTLEY'S METHOD
    else
        // use intrinsic parameters of each camera, but
        // compute the rectification transformation directly
        // from the fundamental matrix
    {
        std::vector<cv::Point2f> allimgpt[2];
        for (k = 0; k < 2; k++)
        {
            for (i = 0; i < cp; i++)
                std::copy(imagePoints[k][i].begin(), imagePoints[k][i].end(), back_inserter(allimgpt[k]));
        }
        F = findFundamentalMat(cv::Mat(allimgpt[0]), cv::Mat(allimgpt[1]), cv::FM_8POINT, 0, 0);
        cv::Mat H1, H2;
        stereoRectifyUncalibrated(cv::Mat(allimgpt[0]), cv::Mat(allimgpt[1]), F, imageSize, H1, H2, 3);

        R1 = cameraMatrix[0].inv()*H1*cameraMatrix[0];
        R2 = cameraMatrix[1].inv()*H2*cameraMatrix[1];
        P1 = cameraMatrix[0];
        P2 = cameraMatrix[1];
    }

    //Precompute maps for cv::remap()
    initUndistortRectifyMap(cameraMatrix[0], cv::Mat(), R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
    initUndistortRectifyMap(cameraMatrix[1], cv::Mat(), R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

    cv::Mat canvas;
    double sf;
    int w, h;
    if (!isVerticalStereo)
    {
        sf = 600. / MAX(imageSize.width, imageSize.height);
        w = cvRound(imageSize.width*sf);
        h = cvRound(imageSize.height*sf);
        canvas.create(h, w * 2, CV_8UC3);
    }
    else
    {
        sf = 300. / MAX(imageSize.width, imageSize.height);
        w = cvRound(imageSize.width*sf);
        h = cvRound(imageSize.height*sf);
        canvas.create(h * 2, w, CV_8UC3);
    }

    for (i = 0; i < 1; i++)
    {
        for (k = 0; k < 2; k++)
        {
            cv::Mat img, rimg, cimg;
            if (k == 0)
            {
                img = cv::imread("./Images/Camera1/2018.png", 0);
            }
            else
            {
                img = cv::imread("./Images/Camera2/2018.png", 0);
            }
            cv::remap(img, rimg, rmap[k][0], rmap[k][1], cv::INTER_CUBIC);
            cv::cvtColor(rimg, cimg, cv::COLOR_GRAY2BGR);
            cv::imwrite("m.bmp", rimg);

            cv::Mat canvasPart = !isVerticalStereo ? canvas(cv::Rect(w*k, 0, w, h)) : canvas(cv::Rect(0, h*k, w, h));
            cv::resize(cimg, canvasPart, canvasPart.size(), 0, 0, cv::INTER_AREA);
            if (useCalibrated)
            {
                cv::Rect vroi(cvRound(validRoi[k].x*sf), cvRound(validRoi[k].y*sf),
                          cvRound(validRoi[k].width*sf), cvRound(validRoi[k].height*sf));
                cv::rectangle(canvasPart, vroi, cv::Scalar(0, 0, 255), 3, 8);
            }
        }

        if (!isVerticalStereo)
            for (j = 0; j < canvas.rows; j += 16)
                line(canvas, cv::Point(0, j), cv::Point(canvas.cols, j), cv::Scalar(0, 255, 0), 1, 8);
        else
            for (j = 0; j < canvas.cols; j += 16)
                line(canvas, cv::Point(j, 0), cv::Point(j, canvas.rows), cv::Scalar(0, 255, 0), 1, 8);
        string file_name;
        file_name = std::string("./ymlo/rectified_").append(std::to_string(cp)).append(std::string("_")).append(std::to_string(i)).append(".bmp");
        imwrite(file_name, canvas);
    }

    std::string other = std::string("./ymlo/other").append(std::to_string(cp)).append(std::string(".yml"));
    fs.open(other, cv::FileStorage::WRITE);
    if (fs.isOpened())
    {
        fs << "remap1 " << rmap[0][0] << "remap2 " << rmap[0][1] << "remap3 " << rmap[1][0] << "remap4 " << rmap[1][1] ;
        fs.release();
    }
    else
        cout << "Error: can not save the other parameters\n";

    return true;
}/**/



static
bool stereo_calibration_routine_ransac(string calibration_path_files,int wind_size, cv::Mat cameraMatrix[2], cv::Mat distCoeffs[2],
std::vector<string> &left_images, std::vector<string> &right_images,
cv::Mat &R, cv::Mat &T, cv::Mat &E, cv::Mat &F, cv::Mat &Q,
Mode_To_Revtreive_Files mode, Type_Of_Calibration calib_type, USE_RECTIFIED_3D_POINTS_GRID rect_grid, string inputFile = "",
double error_rms = 1, double error_avg = 1)
{
    cout << "stereo_calibration_routine_ransac " << endl;

    string results_dir_name = string(".\\yml_").append(to_string(wind_size)).append("\\");

    create_directory(results_dir_name);

    std::vector<string> imageListCam1, imageListCam2;
    typedef std::vector<cv::Point2f> prise;
    typedef std::vector<std::vector<prise>> liste_prise_stereo;
    string inputFilename;

    if (mode == DIRECTORY)
    {

        string left_path = string("..\\").append(calibration_path_files).append("\\Camera1");
        string right_path = string("..\\").append(calibration_path_files).append("\\Camera2");

        imageListCam1 = get_list_of_files(left_path);
        imageListCam2 = get_list_of_files(right_path);
    }
    else if (mode == JSFILES)
    {
        if (inputFile.length() == 0)
        {
            inputFilename = std::string("imageList.old.json");// fichier par défaut
        }
        else
            inputFilename = inputFile;

        remplirListes(inputFilename, imageListCam1, imageListCam2);
    }

    bool displayCorners = false;
    bool useCalibrated = true;
    bool showRectified = true;
    cv::Size boardSize, imageSize;
    float squareSize;

    //Mat cameraMatrix[2], distCoeffs[2];
    string outputFilename;

    int flags = 0;

    liste_prise_stereo imagePoints_C;
    std::vector<std::vector<cv::Point2f> > imagePoints[2];

    std::vector<string> goodImageList1;
    std::vector<string> goodImageList2;
    Pattern pattern;

    boardSize.width = 6;
    boardSize.height = 6;
    pattern = CIRCLES_GRID;
    squareSize = 160;
    outputFilename = std::string("out_stereo_camera_data.json");
    flags |=  cv::CALIB_FIX_ASPECT_RATIO ;
    flags |=  cv::CALIB_FIX_PRINCIPAL_POINT;
    flags |= cv::CALIB_USE_INTRINSIC_GUESS ;
    flags |= cv::CALIB_ZERO_TANGENT_DIST;



    int i, j, k, nimages;

    // extrait les noms des images a partir du fichier json
    //
    nimages = imageListCam1.size();



    if(nimages <= 2)
        return 0;

    assert(nimages >= 2 && "The calibration process needs at least 2 images pairs");

    assert(imageListCam1.size() == imageListCam2.size() && "The number of images are not the same");

    bool extrac_ok;
    int number_points = boardSize.height*boardSize.width;
    int nb_ok = 0;

    imagePoints_C.clear();
    imagePoints_C.resize(nimages);

    for (int i = 0; i < nimages; i++)
    {
        imagePoints_C[i].resize(2);
    }



    cv::FileStorage fs_pt("points.yml", cv::FileStorage::WRITE);

    ofstream image1X;
    image1X.open("points1X.txt");
    ofstream image1Y;
    image1Y.open("points1Y.txt");

    ofstream image2X;
    image2X.open("pointsCX.txt");
    ofstream image2Y;
    image2Y.open("pointsCY.txt");

    int idx_ok = 0;

    for (int i = 0; i < (int)imageListCam1.size(); i++)
    {

        cout << "Images " << imageListCam1[i]  << "  " << imageListCam2[i] << endl;

        cv::Mat view = cv::imread(imageListCam1[i], CV_8U);
        //cv::resize(view, view, cv::Size(), 4, 4 , INTER_CUBIC );
        std::vector<cv::Point2f> im_point1;

        extrac_ok = extrairePointsImage(view, boardSize, im_point1, imageListCam1[i], wind_size);
        if (!extrac_ok)
        {
            cout << "not ok pair 1" << endl;
            continue;
        }
        cout << "Image pair number : " << i << endl << endl;
        //cout << endl << endl << endl;

        cv::Mat view1 = cv::imread(imageListCam2[i], CV_8U);
        //cv::resize(view1, view1, cv::Size(), 4, 4 , INTER_CUBIC );
        std::vector<cv::Point2f> im_point2;
        extrac_ok = extrairePointsImage(view1, boardSize, im_point2, imageListCam2[i], wind_size);

        if (!extrac_ok)
        {
            cout << "not ok pair 2 " << endl;
            continue;
        }

        imagePoints_C[idx_ok][0] = im_point1;
        imagePoints_C[idx_ok][1] = im_point2;
        goodImageList1.push_back(imageListCam1[i]);
        goodImageList2.push_back(imageListCam2[i]);
        nb_ok++;
        idx_ok++;
        imageSize = view.size();
        imwrite(std::string("../ImagesTemp/left_view_").append(to_string(i+1)).append(".png"), view);
        imwrite(std::string("../ImagesTemp/right_view_").append(to_string(i+1)).append(".png"), view1);
    }

    for (int i = 0; i < nb_ok; i++)
    {
        cv::Mat img1, img2;
        img1 = cv::Mat::zeros(imageSize, CV_8U);
        img2 = cv::Mat::zeros(imageSize, CV_8U);
        std::vector<cv::Point2f> im_pt1 = imagePoints_C[i][0];
        std::vector<cv::Point2f> im_pt2 = imagePoints_C[i][1];
        int nb_points = im_pt1.size();
        for (int k = 0; k < nb_points; k++)
        {
            int x, y;
            img1.at<uchar>(im_pt1[k]) = 255;
            img2.at<uchar>(im_pt2[k]) = 255;
        }
        string file_name;
        file_name = std::string("./Images/imageleft").append(std::to_string(i)).append(".bmp");
        left_images.push_back(file_name);
        //imwrite(file_name, img1);
        file_name = std::string("./Images/imageright").append(std::to_string(i)).append(".bmp");
        //imwrite(file_name, img2);
        right_images.push_back(file_name);
    }

    nimages = nb_ok;

    assert(nimages >= 2 && "The calibration process needs at least 2 images pairs");

    (imagePoints_C).resize(nimages);

    cout << "nimages " << nimages << endl;

    //cout << "ni " << 10 << endl;

    double curr_min_rms = 100000;
    double curr_min_avg = 100000;

    int cp = 35;
    int nb_iter = 50;
    int iter = 0;
    if(nimages > 2)
    for (iter = 0; iter < nb_iter; iter++)
    {
        std::vector<int> indices(nimages, 0);

        std::vector<int> taken(cp);

        generate(begin(indices), end(indices), rangegenerator(0));

        std::random_device rd;
        std::mt19937 g(rd());

        //random_shuffle(begin(indices), end(indices));

        std::shuffle(indices.begin(), indices.end(), g);

        std::vector<std::vector<cv::Point3f>> objectPoints;

        objectPoints.resize(cp);

        imagePoints[0].clear();
        imagePoints[1].clear();
        imagePoints[0].resize(cp);
        imagePoints[1].resize(cp);

        //cout << "first loop cp = " << cp << "   " << iter << " " << indices.size() << endl;

        for (int i = 0; i < cp; i++)
        {
            //cout << indices[i] << endl;
            taken[i] = indices[i];
            imagePoints[0][i] = imagePoints_C[indices[i]][0];
            imagePoints[1][i] = imagePoints_C[indices[i]][1];
        }

        CalculateObjectsPoints_Stereo(boardSize, squareSize, objectPoints, cp, pattern);

        cout << "------------------------------------------------------------------------------------------------------------------" << endl;
        cout << "initialization " << endl;

        cameraMatrix[0] = initCameraMatrix2D(objectPoints, imagePoints[0], imageSize, 0);
        cameraMatrix[1] = initCameraMatrix2D(objectPoints, imagePoints[1], imageSize, 0);

        cout << "end of initialization " << endl;

        cout << "start calibration " << endl;


        double rms = stereoCalibrate(objectPoints, (imagePoints[0]), (imagePoints[1]),
                cameraMatrix[0], distCoeffs[0],
                cameraMatrix[1], distCoeffs[1], imageSize, R, T, E, F,
            cv::CALIB_FIX_ASPECT_RATIO +
            cv::CALIB_ZERO_TANGENT_DIST +
            cv::CALIB_SAME_FOCAL_LENGTH +
            cv::CALIB_RATIONAL_MODEL +
            cv::CALIB_FIX_K3 + cv::CALIB_FIX_K4 + cv::CALIB_FIX_K5,
            cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 500, 1e-9)
                );


        //attention aux flags!
        //enregistrement des parametres

        std::cout << "end of callibration" << std:: endl;
        std::cout << "Calibration results : " << std::endl;

        //system("PAUSE");

        // CALIBRATION QUALITY CHECK
        // because the output fundamental matrix implicitly
        // includes all the output information,
        // we can check the quality of calibration using the
        // epipolar geometry constraint: m2^t*F*m1=0

        double err = 0;
        int npoints = 0;
        std::vector<cv::Vec3f> lines[2];
        for (i = 0; i < cp; i++)
        {
            int npt = (int)imagePoints[0][i].size();
            cv::Mat imgpt[2];
            for (k = 0; k < 2; k++)
            {
                imgpt[k] = cv::Mat(imagePoints[k][i]);
                cv::undistortPoints(imgpt[k], imgpt[k], cameraMatrix[k], distCoeffs[k], cv::Mat(), cameraMatrix[k]);
                cv::computeCorrespondEpilines(imgpt[k], k + 1, F, lines[k]);
            }
            for (j = 0; j < npt; j++)
            {
                double errij = fabs(imagePoints[0][i][j].x*lines[1][j][0] +
                        imagePoints[0][i][j].y*lines[1][j][1] + lines[1][j][2]) +
                        fabs(imagePoints[1][i][j].x*lines[0][j][0] +
                        imagePoints[1][i][j].y*lines[0][j][1] + lines[0][j][2]);
                err += errij;
            }
            npoints += npt;
        }

        double avg_epi = err / npoints;


        std::cout << "-----rms = " << rms  << std::endl;

        std::cout << "-----average epipolar error = " << avg_epi << "  points " << npoints << std::endl;

        bool to_continue = false;

        if(rms < error_rms && avg_epi < error_avg)
        {
            cout << "less than threshold" << endl;
            if(curr_min_rms > rms && curr_min_avg > avg_epi )
            {
                cout << "less than min " << endl;
                curr_min_rms = rms;
                curr_min_avg = avg_epi;
                to_continue = true;
            }
            else {
                to_continue = false;
            }
        }
        else
        {
            to_continue = false;
        }



        /*if (rms > 1 && rms < 0.1)
            {
                cout << "tooooo big" << endl << endl << endl;
                continue;
            }*/

        if(to_continue)
        {

            std::cout << "Robust calibration"  << std::endl;

            string temp_results_dir_name = results_dir_name;

            std::string intrinsic = temp_results_dir_name.append(std::string("intrinsics")).append(std::to_string(cp)).append(std::string("_")).append(std::to_string(iter)).append(std::string(".yml"));

            // save intrinsic parameters
            cv::FileStorage fs(intrinsic, cv::FileStorage::WRITE);
            if (fs.isOpened())
            {
                fs << "M1" << cameraMatrix[0] << "D1" << distCoeffs[0] <<
                      "M2" << cameraMatrix[1] << "D2" << distCoeffs[1];
                fs.release();
            }
            else
                cout << "Error: can not save the intrinsic parameters\n";

            cv::Mat R1, R2, P1, P2;
            cv::Rect validRoi[2];

            //stereoRectify(cameraMatrix[0], distCoeffs[0],
            //cameraMatrix[1], distCoeffs[1],
            //imageSize, R, T, R1, R2, P1, P2, Q);

            cv::stereoRectify(cameraMatrix[0], distCoeffs[0],
                    cameraMatrix[1], distCoeffs[1],
                    imageSize, R, T, R1, R2, P1, P2, Q,
                cv::CALIB_ZERO_DISPARITY, 1, imageSize, &validRoi[0], &validRoi[1]);



            if (USE_RECTIFIED_3D_POINTS_GRID::YES == rect_grid)
            {
                stereo_params_cv stereo_par;
                stereo_par.retreive_values();
                cv::Mat _3D_points;

                plane pl;

                cv::Point2f l = imagePoints[0][0][10];
                cv::Point2f r = imagePoints[1][0][10];
                std::vector<cv::Point2f> object_points_left;
                std::vector<cv::Point2f> object_points_right;

                //cout << "left " << l << " right " << r << endl;

                augment_data_around_point(l,r,3,object_points_left,object_points_right);

                //points_3D_reconstruction_rectified_(F, cameraMatrix[0], cameraMatrix[1], P1, P2, R1, R2, T, distCoeffs[0], distCoeffs[1],
                //    imagePoints[0][0], imagePoints[1][0],_3D_points);

                points_3D_reconstruction_rectified(stereo_par, object_points_left, object_points_right, _3D_points, pl, FIT_PLANE_CLOUD_POINT::FIT);

                save_3D_points(_3D_points);

            }

            temp_results_dir_name = results_dir_name;

            std::string extrinsic = temp_results_dir_name.append(std::string("extrinsics_")).append(std::to_string(cp)).append(std::string("_")).append(std::to_string(iter)).append(std::string(".yml"));

            fs.open(extrinsic, cv::FileStorage::WRITE);
            if (fs.isOpened())
            {
                fs << "R" << R << "T" << T << "T norme" << cv::norm(T) << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q << "F" << F << "E" << E
                   << "list_images" << taken << "average_epipolar_error" << err / npoints << "rms" << rms;
                fs.release();
            }
            else
                cout << "Error: can not save the extrinsic parameters\n";



            // OpenCV can handle left-right
            // or up-down camera arrangements
            bool isVerticalStereo = false;

            // COMPUTE AND DISPLAY RECTIFICATION
            if (!showRectified)
                return true;

            cv::Mat rmap[2][2];
            // IF BY CALIBRATED (BOUGUET'S METHOD)
            if (useCalibrated)
            {
                // we already computed everything
            }
            // OR ELSE HARTLEY'S METHOD
            else
                // use intrinsic parameters of each camera, but
                // compute the rectification transformation directly
                // from the fundamental matrix
            {
                std::vector<cv::Point2f> allimgpt[2];
                for (k = 0; k < 2; k++)
                {
                    for (i = 0; i < cp; i++)
                        std::copy(imagePoints[k][i].begin(), imagePoints[k][i].end(), back_inserter(allimgpt[k]));
                }
                F = findFundamentalMat(cv::Mat(allimgpt[0]), cv::Mat(allimgpt[1]), cv::FM_8POINT, 0, 0);
                cv::Mat H1, H2;
                cv::stereoRectifyUncalibrated(cv::Mat(allimgpt[0]), cv::Mat(allimgpt[1]), F, imageSize, H1, H2, 3);

                R1 = cameraMatrix[0].inv()*H1*cameraMatrix[0];
                R2 = cameraMatrix[1].inv()*H2*cameraMatrix[1];
                P1 = cameraMatrix[0];
                P2 = cameraMatrix[1];
            }

            //Precompute maps for cv::remap()
            initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
            initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

            cv::Mat canvas;
            double sf;
            int w, h;
            if (!isVerticalStereo)
            {
                sf = 600. / MAX(imageSize.width, imageSize.height);
                w = cvRound(imageSize.width*sf);
                h = cvRound(imageSize.height*sf);
                canvas.create(h, w * 2, CV_8UC3);
            }
            else
            {
                sf = 300. / MAX(imageSize.width, imageSize.height);
                w = cvRound(imageSize.width*sf);
                h = cvRound(imageSize.height*sf);
                canvas.create(h * 2, w, CV_8UC3);
            }

            for (i = 0; i < cp; i++)
            {
                for (k = 0; k < 2; k++)
                {
                    cv::Mat img, rimg, cimg;
                    if (k == 0)
                    {
                        img = cv::imread(left_images[i], 0);
                    }
                    else
                    {
                        img = cv::imread(right_images[i], 0);
                    }
                    cv::remap(img, rimg, rmap[k][0], rmap[k][1], cv::INTER_LANCZOS4);
                    string file_na;
                    file_na = std::string("./Images/recti_").append(std::to_string(cp)).append(std::string("_")).append(std::to_string(k))
                            .append(std::string("_")).append(std::to_string(i)).append(".bmp");
                    cv::cvtColor(rimg, cimg, cv::COLOR_GRAY2BGR);
                    //imwrite(file_na, rimg);
                    file_na = std::string("./Images/recti__").append(std::to_string(cp)).append(std::string("_")).append(std::to_string(k))
                            .append(std::string("_")).append(std::to_string(i)).append(".bmp");
                    //imwrite(file_na, img);
                    cv::Mat canvasPart = !isVerticalStereo ? canvas(cv::Rect(w*k, 0, w, h)) : canvas(cv::Rect(0, h*k, w, h));
                    resize(cimg, canvasPart, canvasPart.size(), 0, 0, cv::INTER_AREA);
                    if (useCalibrated)
                    {
                        cv::Rect vroi(cvRound(validRoi[k].x*sf), cvRound(validRoi[k].y*sf),
                                  cvRound(validRoi[k].width*sf), cvRound(validRoi[k].height*sf));
                        cv::rectangle(canvasPart, vroi, cv::Scalar(0, 0, 255), 3, 8);
                    }
                }

                if (!isVerticalStereo)
                    for (j = 0; j < canvas.rows; j += 16)
                        line(canvas, cv::Point(0, j), cv::Point(canvas.cols, j), cv::Scalar(0, 255, 0), 1, 8);
                else
                    for (j = 0; j < canvas.cols; j += 16)
                        line(canvas, cv::Point(j, 0), cv::Point(j, canvas.rows), cv::Scalar(0, 255, 0), 1, 8);
                string file_name;
                file_name = std::string("./Images/rectified_").append(std::to_string(cp)).append(std::string("_")).append(std::to_string(i)).append(".bmp");
                //imwrite(file_name, canvas);
                //imshow("rectified", canvas);
                //char c = (char)waitKey();
                //if (c == 27 || c == 'q' || c == 'Q')
                //	break;
            }/**/

        }

        cout << "------------------------------------------------------------------------------------------------------------------" << endl;

        cout << endl << endl << endl;







        //return true;


    }/**/

    return true;
}


static
bool stereo_calibration_routine_ransac_all(cv::Mat cameraMatrix[2], cv::Mat distCoeffs[2],
std::vector<string> &left_images, std::vector<string> &right_images,
cv::Mat &R, cv::Mat &T, cv::Mat &E, cv::Mat &F, cv::Mat &Q,
Mode_To_Revtreive_Files mode, Type_Of_Calibration calib_type, USE_RECTIFIED_3D_POINTS_GRID rect_grid, string inputFile = "",
double error_rms = 1, double error_avg = 1)
{
    cout << "stereo_calibration_routine_ransac " << endl;


    std::vector<string> imageListCam1, imageListCam2;
    typedef std::vector<cv::Point2f> prise;
    typedef std::vector<std::vector<prise>> liste_prise_stereo;
    string inputFilename;

    if (mode == DIRECTORY)
    {
        imageListCam1 = get_list_of_files("..\\CalibrationGold\\Camera1");
        imageListCam2 = get_list_of_files("..\\CalibrationGold\\Camera2");
    }
    else if (mode == JSFILES)
    {
        if (inputFile.length() == 0)
        {
            inputFilename = std::string("imageList.old.json");// fichier par défaut
        }
        else
            inputFilename = inputFile;

        remplirListes(inputFilename, imageListCam1, imageListCam2);
    }

    bool displayCorners = false;
    bool useCalibrated = true;
    bool showRectified = true;
    cv::Size boardSize, imageSize;
    float squareSize;

    //Mat cameraMatrix[2], distCoeffs[2];
    string outputFilename;

    int flags = 0;

    liste_prise_stereo imagePoints_C;
    std::vector<std::vector<cv::Point2f> > imagePoints[2];

    std::vector<string> goodImageList1;
    std::vector<string> goodImageList2;
    Pattern pattern;

    boardSize.width = 6;
    boardSize.height = 6;
    pattern = CIRCLES_GRID;
    squareSize = 160;
    outputFilename = std::string("out_stereo_camera_data.json");
    flags |=  cv::CALIB_FIX_ASPECT_RATIO ;
    flags |=  cv::CALIB_FIX_PRINCIPAL_POINT;
    flags |= cv::CALIB_USE_INTRINSIC_GUESS ;
    flags |= cv::CALIB_ZERO_TANGENT_DIST;



    int i, j, k, nimages;

    // extrait les noms des images a partir du fichier json
    //
    nimages = imageListCam1.size();



    if(nimages <= 2)
        return 0;

    assert(nimages >= 2 && "The calibration process needs at least 2 images pairs");

    assert(imageListCam1.size() == imageListCam2.size() && "The number of images are not the same");

    bool extrac_ok;
    int number_points = boardSize.height*boardSize.width;
    int nb_ok = 0;

    imagePoints_C.clear();
    imagePoints_C.resize(nimages);

    for (int i = 0; i < nimages; i++)
    {
        imagePoints_C[i].resize(2);
    }



    cv::FileStorage fs_pt("points.yml", cv::FileStorage::WRITE);

    ofstream image1X;
    image1X.open("points1X.txt");
    ofstream image1Y;
    image1Y.open("points1Y.txt");

    ofstream image2X;
    image2X.open("pointsCX.txt");
    ofstream image2Y;
    image2Y.open("pointsCY.txt");

    int idx_ok = 0;

    for (int i = 0; i < (int)imageListCam1.size(); i++)
    {

        cout << "Images " << imageListCam1[i]  << "  " << imageListCam2[i] << endl;

        cv::Mat view = cv::imread(imageListCam1[i], CV_16U);
        cv::resize(view, view, cv::Size(), 4, 4 , cv::INTER_LANCZOS4 );
        std::vector<cv::Point2f> im_point1;

        extrac_ok = extrairePointsImage(view, boardSize, im_point1, imageListCam1[i]);
        if (!extrac_ok)
        {
            cout << "not ok pair 1" << endl;
            continue;
        }
        cout << "Image pair number : " << i << endl << endl;
        //cout << endl << endl << endl;

        cv::Mat view1 = cv::imread(imageListCam2[i], CV_16U);
        cv::resize(view1, view1, cv::Size(), 4, 4 , cv::INTER_LANCZOS4 );
        std::vector<cv::Point2f> im_point2;
        extrac_ok = extrairePointsImage(view1, boardSize, im_point2, imageListCam2[i]);

        if (!extrac_ok)
        {
            cout << "not ok pair 2 " << endl;
            continue;
        }

        imagePoints_C[idx_ok][0] = im_point1;
        imagePoints_C[idx_ok][1] = im_point2;
        goodImageList1.push_back(imageListCam1[i]);
        goodImageList2.push_back(imageListCam2[i]);
        nb_ok++;
        idx_ok++;
        imageSize = view.size();
        imwrite(std::string("../ImagesTemp/left_view_").append(to_string(i+1)).append(".png"), view);
        imwrite(std::string("../ImagesTemp/right_view_").append(to_string(i+1)).append(".png"), view1);
    }

    for (int i = 0; i < nb_ok; i++)
    {
        cv::Mat img1, img2;
        img1 = cv::Mat::zeros(imageSize, CV_8U);
        img2 = cv::Mat::zeros(imageSize, CV_8U);
        std::vector<cv::Point2f> im_pt1 = imagePoints_C[i][0];
        std::vector<cv::Point2f> im_pt2 = imagePoints_C[i][1];
        int nb_points = im_pt1.size();
        for (int k = 0; k < nb_points; k++)
        {
            int x, y;
            img1.at<uchar>(im_pt1[k]) = 255;
            img2.at<uchar>(im_pt2[k]) = 255;
        }
        string file_name;
        file_name = std::string("./Images/imageleft").append(std::to_string(i)).append(".bmp");
        left_images.push_back(file_name);
        //imwrite(file_name, img1);
        file_name = std::string("./Images/imageright").append(std::to_string(i)).append(".bmp");
        //imwrite(file_name, img2);
        right_images.push_back(file_name);
    }

    nimages = nb_ok;

    assert(nimages >= 2 && "The calibration process needs at least 2 images pairs");

    (imagePoints_C).resize(nimages);

    //cout << "taille " << (imagePoints[0]).size() << "   " << (imagePoints[1]).size() << " nimages " << nimages << endl;

    //assert((imagePoints[0]).size() == (imagePoints[1]).size() && (imagePoints[0]).size() == nimages && "The extraction is not well performed");

    //system("PAUSE");

    //cout << "taille " << imagePoints_C[0][0].size() << endl;

    //cout << "objectPoints " << objectPoints.size() << endl;


    cout << "nimages " << nimages << endl;

    //cout << "ni " << 10 << endl;

    double curr_min_rms = 100000;
    double curr_min_avg = 100000;

    int cp = 5;
    int nb_iter = 10;
    int iter = 0;
    if(nimages > 2)
    for (iter = 0; iter < nb_iter; iter++)
    {
        std::vector<int> indices(nimages, 0);

        std::vector<int> taken(cp);

        generate(begin(indices), end(indices), rangegenerator(0));

        std::random_device rd;
        std::mt19937 g(rd());

        //random_shuffle(begin(indices), end(indices));

        std::shuffle(indices.begin(), indices.end(), g);

        std::vector<std::vector<cv::Point3f>> objectPoints;

        objectPoints.resize(cp);

        imagePoints[0].clear();
        imagePoints[1].clear();
        imagePoints[0].resize(cp);
        imagePoints[1].resize(cp);

        //cout << "first loop cp = " << cp << "   " << iter << " " << indices.size() << endl;

        for (int i = 0; i < cp; i++)
        {
            //cout << indices[i] << endl;
            taken[i] = indices[i];
            imagePoints[0][i] = imagePoints_C[indices[i]][0];
            imagePoints[1][i] = imagePoints_C[indices[i]][1];
        }

        CalculateObjectsPoints_Stereo(boardSize, squareSize, objectPoints, cp, pattern);

        cout << "------------------------------------------------------------------------------------------------------------------" << endl;
        cout << "initialization " << endl;

        cameraMatrix[0] = initCameraMatrix2D(objectPoints, imagePoints[0], imageSize, 0);
        cameraMatrix[1] = initCameraMatrix2D(objectPoints, imagePoints[1], imageSize, 0);

        cout << "end of initialization " << endl;

        cout << "start calibration " << endl;


        double rms = stereoCalibrate(objectPoints, (imagePoints[0]), (imagePoints[1]),
                cameraMatrix[0], distCoeffs[0],
                cameraMatrix[1], distCoeffs[1], imageSize, R, T, E, F,
            cv::CALIB_FIX_ASPECT_RATIO +
            cv::CALIB_ZERO_TANGENT_DIST +
            cv::CALIB_SAME_FOCAL_LENGTH +
            cv::CALIB_RATIONAL_MODEL +
            cv::CALIB_FIX_K3 + cv::CALIB_FIX_K4 + cv::CALIB_FIX_K5,
            cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 500, 1e-9)
                );


        //attention aux flags!
        //enregistrement des parametres

        cout << "end of callibration" << endl;
        cout << "Calibration results : " << endl;

        //system("PAUSE");

        // CALIBRATION QUALITY CHECK
        // because the output fundamental matrix implicitly
        // includes all the output information,
        // we can check the quality of calibration using the
        // epipolar geometry constraint: m2^t*F*m1=0

        double err = 0;
        int npoints = 0;
        std::vector<cv::Vec3f> lines[2];
        for (i = 0; i < cp; i++)
        {
            int npt = (int)imagePoints[0][i].size();
            cv::Mat imgpt[2];
            for (k = 0; k < 2; k++)
            {
                imgpt[k] = cv::Mat(imagePoints[k][i]);
                undistortPoints(imgpt[k], imgpt[k], cameraMatrix[k], distCoeffs[k], cv::Mat(), cameraMatrix[k]);
                computeCorrespondEpilines(imgpt[k], k + 1, F, lines[k]);
            }
            for (j = 0; j < npt; j++)
            {
                double errij = fabs(imagePoints[0][i][j].x*lines[1][j][0] +
                        imagePoints[0][i][j].y*lines[1][j][1] + lines[1][j][2]) +
                        fabs(imagePoints[1][i][j].x*lines[0][j][0] +
                        imagePoints[1][i][j].y*lines[0][j][1] + lines[0][j][2]);
                err += errij;
            }
            npoints += npt;
        }

        double avg_epi = err / npoints;


        cout << "-----rms = " << rms  << endl;

        cout << "-----average epipolar error = " << avg_epi << "  points " << npoints << endl;

        bool to_continue = false;

        if(rms < error_rms && avg_epi < error_avg)
        {
            cout << "less than threshold" << endl;
            if(curr_min_rms > rms && curr_min_avg > avg_epi )
            {
                cout << "less than min " << endl;
                curr_min_rms = rms;
                curr_min_avg = avg_epi;
                to_continue = true;
            }
            else {
                to_continue = false;
            }
        }
        else
        {
            to_continue = false;
        }



        /*if (rms > 1 && rms < 0.1)
            {
                cout << "tooooo big" << endl << endl << endl;
                continue;
            }*/

        if(to_continue)
        {

            cout << "Robust calibration"  << endl;

            std::string intrinsic = std::string("./yml_wind_6/intrinsics").append(std::to_string(cp)).append(std::string("_")).append(std::to_string(iter)).append(std::string(".yml"));

            // save intrinsic parameters
            cv::FileStorage fs(intrinsic, cv::FileStorage::WRITE);
            if (fs.isOpened())
            {
                fs << "M1" << cameraMatrix[0] << "D1" << distCoeffs[0] <<
                      "M2" << cameraMatrix[1] << "D2" << distCoeffs[1];
                fs.release();
            }
            else
                cout << "Error: can not save the intrinsic parameters\n";

            cv::Mat R1, R2, P1, P2;
            cv::Rect validRoi[2];

            //stereoRectify(cameraMatrix[0], distCoeffs[0],
            //cameraMatrix[1], distCoeffs[1],
            //imageSize, R, T, R1, R2, P1, P2, Q);

            cv::stereoRectify(cameraMatrix[0], distCoeffs[0],
                    cameraMatrix[1], distCoeffs[1],
                    imageSize, R, T, R1, R2, P1, P2, Q,
                cv::CALIB_ZERO_DISPARITY, 1, imageSize, &validRoi[0], &validRoi[1]);



            if (USE_RECTIFIED_3D_POINTS_GRID::YES == rect_grid)
            {
                stereo_params_cv stereo_par;
                stereo_par.retreive_values();
                cv::Mat _3D_points;

                plane pl;

                cv::Point2f l = imagePoints[0][0][10];
                cv::Point2f r = imagePoints[1][0][10];
                std::vector<cv::Point2f> object_points_left;
                std::vector<cv::Point2f> object_points_right;

                //cout << "left " << l << " right " << r << endl;

                augment_data_around_point(l,r,3,object_points_left,object_points_right);

                //points_3D_reconstruction_rectified_(F, cameraMatrix[0], cameraMatrix[1], P1, P2, R1, R2, T, distCoeffs[0], distCoeffs[1],
                //    imagePoints[0][0], imagePoints[1][0],_3D_points);

                points_3D_reconstruction_rectified(stereo_par, object_points_left, object_points_right, _3D_points, pl, FIT_PLANE_CLOUD_POINT::FIT);

                save_3D_points(_3D_points);

            }

            std::string extrinsic = std::string("./yml_wind_6/extrinsics").append(std::to_string(cp)).append(std::string("_")).append(std::to_string(iter)).append(std::string(".yml"));

            fs.open(extrinsic, cv::FileStorage::WRITE);
            if (fs.isOpened())
            {
                fs << "R" << R << "T" << T << "T norme" << cv::norm(T) << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q << "F" << F << "E" << E
                   << "list_images" << taken << "average_epipolar_error" << err / npoints << "rms" << rms;
                fs.release();
            }
            else
                cout << "Error: can not save the extrinsic parameters\n";



            // OpenCV can handle left-right
            // or up-down camera arrangements
            bool isVerticalStereo = false;

            // COMPUTE AND DISPLAY RECTIFICATION
            if (!showRectified)
                return true;

            cv::Mat rmap[2][2];
            // IF BY CALIBRATED (BOUGUET'S METHOD)
            if (useCalibrated)
            {
                // we already computed everything
            }
            // OR ELSE HARTLEY'S METHOD
            else
                // use intrinsic parameters of each camera, but
                // compute the rectification transformation directly
                // from the fundamental matrix
            {
                std::vector<cv::Point2f> allimgpt[2];
                for (k = 0; k < 2; k++)
                {
                    for (i = 0; i < cp; i++)
                        std::copy(imagePoints[k][i].begin(), imagePoints[k][i].end(), back_inserter(allimgpt[k]));
                }
                F = findFundamentalMat(cv::Mat(allimgpt[0]), cv::Mat(allimgpt[1]), cv::FM_8POINT, 0, 0);
                cv::Mat H1, H2;
                stereoRectifyUncalibrated(cv::Mat(allimgpt[0]), cv::Mat(allimgpt[1]), F, imageSize, H1, H2, 3);

                R1 = cameraMatrix[0].inv()*H1*cameraMatrix[0];
                R2 = cameraMatrix[1].inv()*H2*cameraMatrix[1];
                P1 = cameraMatrix[0];
                P2 = cameraMatrix[1];
            }

            //Precompute maps for cv::remap()
            initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
            initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

            cv::Mat canvas;
            double sf;
            int w, h;
            if (!isVerticalStereo)
            {
                sf = 600. / MAX(imageSize.width, imageSize.height);
                w = cvRound(imageSize.width*sf);
                h = cvRound(imageSize.height*sf);
                canvas.create(h, w * 2, CV_8UC3);
            }
            else
            {
                sf = 300. / MAX(imageSize.width, imageSize.height);
                w = cvRound(imageSize.width*sf);
                h = cvRound(imageSize.height*sf);
                canvas.create(h * 2, w, CV_8UC3);
            }

            for (i = 0; i < cp; i++)
            {
                for (k = 0; k < 2; k++)
                {
                    cv::Mat img, rimg, cimg;
                    if (k == 0)
                    {
                        img = cv::imread(left_images[i], 0);
                    }
                    else
                    {
                        img = cv::imread(right_images[i], 0);
                    }
                    cv::remap(img, rimg, rmap[k][0], rmap[k][1], cv::INTER_LANCZOS4);
                    string file_na;
                    file_na = std::string("./Images/recti_").append(std::to_string(cp)).append(std::string("_")).append(std::to_string(k))
                            .append(std::string("_")).append(std::to_string(i)).append(".bmp");
                    cv::cvtColor(rimg, cimg, cv::COLOR_GRAY2BGR);
                    //imwrite(file_na, rimg);
                    file_na = std::string("./Images/recti__").append(std::to_string(cp)).append(std::string("_")).append(std::to_string(k))
                            .append(std::string("_")).append(std::to_string(i)).append(".bmp");
                    //imwrite(file_na, img);
                    cv::Mat canvasPart = !isVerticalStereo ? canvas(cv::Rect(w*k, 0, w, h)) : canvas(cv::Rect(0, h*k, w, h));
                    cv::resize(cimg, canvasPart, canvasPart.size(), 0, 0, cv::INTER_AREA);
                    if (useCalibrated)
                    {
                        cv::Rect vroi(cvRound(validRoi[k].x*sf), cvRound(validRoi[k].y*sf),
                                  cvRound(validRoi[k].width*sf), cvRound(validRoi[k].height*sf));
                        rectangle(canvasPart, vroi, cv::Scalar(0, 0, 255), 3, 8);
                    }
                }

                if (!isVerticalStereo)
                    for (j = 0; j < canvas.rows; j += 16)
                        line(canvas, cv::Point(0, j), cv::Point(canvas.cols, j), cv::Scalar(0, 255, 0), 1, 8);
                else
                    for (j = 0; j < canvas.cols; j += 16)
                        line(canvas, cv::Point(j, 0), cv::Point(j, canvas.rows), cv::Scalar(0, 255, 0), 1, 8);
                string file_name;
                file_name = std::string("./Images/rectified_").append(std::to_string(cp)).append(std::string("_")).append(std::to_string(i)).append(".bmp");
                //imwrite(file_name, canvas);
                //imshow("rectified", canvas);
                //char c = (char)waitKey();
                //if (c == 27 || c == 'q' || c == 'Q')
                //	break;
            }/**/

        }

        cout << "------------------------------------------------------------------------------------------------------------------" << endl;

        cout << endl << endl << endl;







        //return true;


    }/**/

    return true;
}

}
