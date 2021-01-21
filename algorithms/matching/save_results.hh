#pragma once

#include <Eigen/Dense>
#include <Eigen/Core>
#include <list>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "core.hpp"
#include "algorithms/miscellaneous.hh"


#ifdef vpp
#include <vpp/vpp.hh>
#include <vpp/utils/opencv_bridge.hh>
#include <vpp/utils/opencv_utils.hh>
using namespace vpp;
#endif

 
using namespace Eigen;


namespace tpp {



void save_stereo_images(MatrixXf left_image, MatrixXf right_image, cv::Mat &im_p_all,float eps,std::vector<stereo_match> epipolar_order_constraints,int cp = 0,
                        int shift_rows = 0, int shift_cols = 0)
{

    int ncols = left_image.cols();
    int nrows = left_image.rows();

    //get location of maximum
    MatrixXf::Index maxRow_l, maxCol_l;
    float max_l = left_image.maxCoeff(&maxRow_l, &maxCol_l);
    //get location of minimum
    MatrixXf::Index minRow_l, minCol_l;
    float min_l = left_image.minCoeff(&minRow_l, &minCol_l);

    //get location of maximum
    MatrixXf::Index maxRow_r, maxCol_r;
    float max_r = right_image.maxCoeff(&maxRow_r, &maxCol_r);
    //get location of minimum
    MatrixXf::Index minRow_r, minCol_r;
    float min_r = right_image.minCoeff(&minRow_r, &minCol_r);

    uchar min_val = uchar(eps * 255 / max_r);


    for (int rl = 0; rl < nrows; rl++)
    {
        for (int cl = 0; cl < ncols; cl++)
        {
            uchar val = uchar(right_image(rl, cl) * 255 / max_r);
            //if(val > 0)
            {
                im_p_all.at<Vec3b>(rl + shift_rows, cl + shift_cols)[0] = val;
                im_p_all.at<Vec3b>(rl + shift_rows, cl + shift_cols)[1] = val;
                im_p_all.at<Vec3b>(rl + shift_rows, cl + shift_cols)[2] = val;
            }
        }
    }

    for (int rl = 0; rl < nrows; rl++)
    {
        for (int cl = 0; cl < ncols; cl++)
        {
            uchar val = uchar(left_image(rl, cl) * 255 / max_l);
            //if(val > 0)
            {
                im_p_all.at<Vec3b>(rl, cl)[0] = val;
                im_p_all.at<Vec3b>(rl, cl)[1] = val;
                im_p_all.at<Vec3b>(rl, cl)[2] = val;
            }
        }
    }

    int nb = 0;
    for (auto &smt : epipolar_order_constraints)
    {
        Mat temp_mat = im_p_all.clone();
        //stereo_match smt = l;
        if( fabs(smt.first_point.y() - smt.second_point.y()) < 10)
        {

            if (smt.similarity <= 0.5)
                break;

            cout << "first  [ y = " << smt.first_point.y()  << " ; x = " << smt.first_point.x() << " ]  ----   second [ y = "
                 << smt.second_point.y() << " ; x = " << smt.second_point.x() << " ]   similarity " << smt.similarity << endl;
            Point pt1;
            pt1.y = smt.first_point.y();
            pt1.x = smt.first_point.x();
            Point pt2;
            pt2.y = smt.second_point.y() + shift_rows;
            pt2.x = smt.second_point.x() + shift_cols;

            Point pt_;
            pt_.y = smt.second_point.y();
            pt_.x = smt.second_point.x();
            Vector3i color = generate_color(500);

            cv::line(temp_mat, pt1, pt2, cv::Scalar(color[0], color[1], color[2]), 1);
            //cv::circle(disp, pt1, 1, cv::Scalar(color[0], color[1], color[2]), 1);
            nb++;
            //cout << "similarities " << smt.similarity << endl;
            std::string file_name = std::string("../Matching//");
            file_name = file_name.append(std::to_string(cp)).append(std::string("_")).append(std::to_string(nb)).append(std::string("_sim.bmp"));
            imwrite(file_name, temp_mat);
            if (nb > 100000)
                break;
        }
    }


}



void save_stereo_images_part(MatrixXf left_image, MatrixXf right_image, float eps, std::vector<stereo_match> epipolar_order_constraints,int cp=0)
{

    //cout << "save_stereo_images_part"  << endl;

    int ncols = left_image.cols();
    int nrows = left_image.rows();

    //get location of maximum
    MatrixXf::Index maxRow_l, maxCol_l;
    float max_l = left_image.maxCoeff(&maxRow_l, &maxCol_l);
    //get location of minimum
    MatrixXf::Index minRow_l, minCol_l;
    float min_l = left_image.minCoeff(&minRow_l, &minCol_l);

    //get location of maximum
    MatrixXf::Index maxRow_r, maxCol_r;
    float max_r = right_image.maxCoeff(&maxRow_r, &maxCol_r);
    //get location of minimum
    MatrixXf::Index minRow_r, minCol_r;
    float min_r = right_image.minCoeff(&minRow_r, &minCol_r);

    uchar min_val = uchar(eps * 255 / max_r);

    Mat left_mat_ = Mat::zeros(nrows,ncols , CV_8U);
    Mat right_mat_ = Mat::zeros(nrows,ncols , CV_8U);


    for (int rl = 0; rl < nrows; rl++)
    {
        for (int cl = 0; cl < ncols; cl++)
        {
            uchar val = uchar(right_image(rl, cl) * 255 / max_r);
            //if(val > 0)
            {
                right_mat_.at<uchar>(rl , cl ) = val;
            }
        }
    }

    for (int rl = 0; rl < nrows; rl++)
    {
        for (int cl = 0; cl < ncols; cl++)
        {
            uchar val = uchar(left_image(rl, cl) * 255 / max_l);
            //if(val > 0)
            {
                left_mat_.at<uchar>(rl, cl) = val;
            }
        }
    }



    Mat left_mat = Mat::zeros(nrows,ncols , CV_8U);
    Mat right_mat = Mat::zeros(nrows,ncols , CV_8U);


    cvtColor(left_mat_, left_mat, cv::COLOR_GRAY2BGR);
    cvtColor(right_mat_, right_mat, cv::COLOR_GRAY2BGR);

    //cout << "taille " << nrows << "  " << ncols << endl;

    //cout << "taille " << left_mat.rows << "  " << left_mat.cols << endl;

    int nb = 0;
    for (auto &smt : epipolar_order_constraints)
    {
        //stereo_match smt = l;
        //if( fabs(smt.left_point[0] - smt.right_point[0]) < 10 && nb )
        {
            //break;

            if (smt.similarity <= 0.1)
                break;

            //cout << "first  [ y = " << smt.left_point[0]  << " ; x = " << smt.left_point[1] << " ]  ----   second [ y = " << smt.right_point[0] << " ; x = " << smt.right_point[1] << " ]   similarity " << smt.similarity << endl;
            Point pt1;
            pt1.y = smt.first_point.y();
            pt1.x = smt.first_point.x();
            Point pt2;
            pt2.y = smt.second_point.y();
            pt2.x = smt.second_point.x();

            Point pt_;
            pt_.y = smt.second_point.y();
            pt_.x = smt.second_point.x();
            Vector3i color = generate_color(500);

            //if(nb==10)
            {
                left_mat.at<Vec3b>(pt1) = Vec3b(color[0], color[1], color[2]);
                right_mat.at<Vec3b>(pt2) = Vec3b(color[0], color[1], color[2]);
            }




            nb++;
            if (nb > 100000)
                break;
        }
    }

    //cout << "nb "  << nb << " cp " << cp << endl;

    //cv::resize(left_mat, left_mat, cv::Size(), 10, 10 , INTER_CUBIC );
    //cv::resize(right_mat, right_mat, cv::Size(), 10, 10 , INTER_CUBIC );

    std::string file_name_left = std::string("..//Matching//");
    file_name_left = file_name_left.append(std::to_string(cp)).append(std::string("_")).append(std::string("left")).append(std::string("_sim.bmp"));
    imwrite(file_name_left, left_mat);

    std::string file_name_right = std::string("..//Matching//");
    file_name_right = file_name_right.append(std::to_string(cp)).append(std::string("_")).append(std::string("right")).append(std::string("_sim.bmp"));
    imwrite(file_name_right, right_mat);

    //cout << "save_stereo_images_part end"  << endl;


}


}
