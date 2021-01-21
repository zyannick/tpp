#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include "global_var.hh"
#include <experimental/filesystem>
#include "core.hpp"
#include "algorithms/miscellaneous.hh"
#include "global_utils.hh"


using namespace std;
using namespace Eigen;
 

namespace tpp
{


  std::pair<cv::Mat, cv::Mat> format_images(int index_img, string label_path, std::vector<string> imageListCam1, std::vector<string> imageListCam2, int scale_factor )
  {


      string path_at_this_index = string(label_path) + "//_image_" + to_string(index_img) + "_labelled";
      create_directory(path_at_this_index);
      string path_at_this_index_left = path_at_this_index + "//left";
      create_directory(path_at_this_index_left);
      string path_at_this_index_right = path_at_this_index + "//right";
      create_directory(path_at_this_index_right);


      std::vector<Vector3d> list_3d_points;

      cv::Mat left_view_16 = cv::imread(imageListCam1[index_img], CV_16U);
      cv::Mat left_view;

      double minVal, maxVal;
      cv::Point minLoc, maxLoc;
      cv::minMaxLoc(left_view_16, &minVal, &maxVal, &minLoc, &maxLoc);
      // cout << "min val " << minVal << "    max val " << maxVal << endl;
      left_view_16.convertTo(left_view, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));

      if( scale_factor > 1)
          cv::resize(left_view, left_view, cv::Size(), scale_factor, scale_factor , cv::INTER_CUBIC );

      cv::Mat right_view_16 = cv::imread(imageListCam2[index_img], CV_16U);
      cv::Mat right_view;

      cv::minMaxLoc(right_view_16, &minVal, &maxVal, &minLoc, &maxLoc);
      right_view_16.convertTo(right_view, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));

      if( scale_factor > 1)
          cv::resize(right_view, right_view, cv::Size(), scale_factor, scale_factor , cv::INTER_CUBIC );

      return std::pair(left_view, right_view);
  }

void retreive_images_list(int mode, std::vector<string> &left_images, std::vector<string>& right_images, std::string preffix)
{

    string miscellaneous;

    if(mode == 0)
    {
        string inputFilename = miscellaneous;
        inputFilename = inputFilename.append(std::string("./json/depth_4.json"));
        //remplirListes(inputFilename, left_images, right_images);
    }
    else if(mode==1)
    {
        for(int ster = 0 ; ster < 10; ster++)
        {
            string file_src = miscellaneous;
            file_src = file_src.append( std::string("./Images/Persons/_18.025/Camera1/original_").append(std::to_string(ster)).append(".bmp"));
            left_images.push_back(file_src);
            string file_dst = miscellaneous;
            file_dst = file_dst.append(std::string("./Images/Persons/_18.025/Camera2/shift_").append(std::to_string(ster)).append(".bmp"));
            right_images.push_back(file_dst);
        }
    }
}


void down_sample_by(cv::Mat src, cv::Mat &dst, int factor)
{
    int f = factor;
    cv::Mat temp = src;
    while(f >= 1)
    {

        cout << "divsion "  << f << endl;

        pyrDown( temp, dst, cv::Size( temp.cols/2, temp.rows/2 ) );

        cout << "src " << temp.size() << "   dst " << dst.size() << endl;

        temp = dst;
        f = f/2;

    }

}


void addGaussianNoise(cv::Mat &image, double average=0.0, double standard_deviation=10.0)
{

    // We need to work with signed images (as noise can be
    // negative as well as positive). We use 16 bit signed
    // images as otherwise we would lose precision.
    cv::Mat noise_image(image.size(), CV_16SC1);
    randn(noise_image, cv::Scalar::all(average), cv::Scalar::all(standard_deviation));
    //cout << "addGaussianNoise" << endl;
    cv::Mat temp_image;
    image.convertTo(temp_image,CV_16SC1);
    //cout << "addGaussianNoise 1" << endl;
    addWeighted(temp_image, 1.0, noise_image, 1.0, 0.0, temp_image);
    //cout << "addGaussianNoise 2" << endl;
    temp_image.convertTo(image,image.type());
}




string name_from_type_of_features(TYPE_OF_FEATURES tof)
{
    if(TYPE_OF_FEATURES::ORB == tof)
    {
        return "features_labelled_orb";
    }
    else if(TYPE_OF_FEATURES::FAST == tof)
    {
        return "features_labelled_fast";
    }
    else if(TYPE_OF_FEATURES::GFTT == tof)
    {
        return "features_labelled_gftt";
    }
    else if(TYPE_OF_FEATURES::KAZE == tof)
    {
        return "features_labelled_kaze";
    }
    else if(TYPE_OF_FEATURES::SURF == tof)
    {
        return "features_labelled_surf";
    }
    else if(TYPE_OF_FEATURES::AGAST == tof)
    {
        return "features_labelled_agast";
    }
    else if(TYPE_OF_FEATURES::BRISK == tof)
    {
        return "features_labelled_brisk";
    }
    else if(TYPE_OF_FEATURES::SHI_TOMASI == tof)
    {
        return "features_labelled_shi_tomasi";
    }
    else if(TYPE_OF_FEATURES::PHASE_CONGRUENCY == tof)
    {
        return "features_labelled_phase_congruency";
    }
}



}
