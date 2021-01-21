#pragma once

#include "feature_extractor/gabor.hh"

#include "feature_extractor/harris_corners.hh"

#include "feature_extractor/shi_tomasi.hh"

#include "feature_extractor/multiple_ext.hh"

#include "algorithms/generate_maps/utils.hh"

#include <time.h>
#include <chrono>

#include <type_traits>

using namespace std;
using namespace Eigen;
using namespace std::chrono;

namespace tpp
{


float calculateSD(Eigen::ArrayXd  data)
{
    float sum = 0.0, mean, standardDeviation = 0.0;
    int i;
    for(i = 0; i < data.rows(); ++i)
    {
        sum += data(i);
    }
    mean = sum/data.rows();
    for(i = 0; i < data.rows(); ++i)
        standardDeviation += pow(data(i) - mean, 2);
    return sqrt(standardDeviation / data.rows());
}

void feature_detection(cv::Mat rmap[2][2], cv::Rect validRoi[2], cv::Mat img_left, cv::Mat img_right)
{
    cv::Mat canvas;
    cv::Mat rimg_left, cimg_left, rimg_right, cimg_right;
    cv::remap(img_left, rimg_left, rmap[0][0], rmap[0][1], cv::INTER_LINEAR);
    cv::remap(img_left, rimg_left, rmap[1][0], rmap[1][1], cv::INTER_LINEAR);
}

int feature_extractor_main(cv::Mat img_, TYPE_OF_FEATURES tof,
                           std::vector<cv::KeyPoint> &kps, int r = 0 , int beta_ = 0, bool save = false)
{

    int nb_times = 10;

    if(tof == TYPE_OF_FEATURES::HARRIS_CORNERS)
    {
        ///harris
        cv::Mat img_harris = img_.clone();
        harris_detector_settings hst;
        default_initialization_harris_settings(hst);

        std::vector<cv::Point2f> corners_harris;
        //MatrixXf vec = MatrixXf::Zero(1000,1);
        Eigen::ArrayXd vec = ArrayXd::Zero(nb_times, 1);
        for(int i = 0 ; i < nb_times; i++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            extract_corners_harris(img_harris, corners_harris, hst);
            auto end = std::chrono::high_resolution_clock::now();
            vec(i) = duration_cast<microseconds>(end - start).count();
        }

        double std_dev = std::sqrt((vec - vec.mean()).square().sum()/(vec.size()-1));
        cout << "mean " << vec.mean() << endl;
        cout << "stdv " << std_dev << endl;

        return 0;

        std::vector<cv::KeyPoint> keypoint_harris;
        cv::cvtColor(img_harris, img_harris, cv::COLOR_GRAY2BGR);
        for(int i = 0; i < corners_harris.size(); i++)
        {
            keypoint_harris.push_back(cv::KeyPoint(corners_harris[i].x, corners_harris[i].y, 0 ) );
        }
        kps = keypoint_harris;
        cout << "harris" << endl;
        return  corners_harris.size();
    }
    else if(tof == TYPE_OF_FEATURES::SHI_TOMASI)
    {
        //shi and tomasi ///
        cv::Mat img_shi_tomasi = img_.clone();
        cv::cvtColor(img_shi_tomasi, img_shi_tomasi, cv::COLOR_GRAY2BGR);
        shi_tomasi_settings sts;
        default_initialization_shi_tomasi(sts);
        std::vector<cv::Point2f> corners_shi_tomasi;
        extract_shi_tomasi_corners(img_shi_tomasi, corners_shi_tomasi, sts);

        Eigen::ArrayXd vec = ArrayXd::Zero(nb_times+1, 1);
        for(int i = 0 ; i < nb_times+1 ; i++)
        {
            std::vector<cv::Point2f> corners_shi_tomasi;
            auto start = std::chrono::high_resolution_clock::now();
            extract_shi_tomasi_corners(img_shi_tomasi, corners_shi_tomasi, sts);
            auto end = std::chrono::high_resolution_clock::now();
            vec(i) = duration_cast<microseconds>(end - start).count();
        }

        vec = vec.segment(1,nb_times);

        cout << vec << endl << endl;

        double std_dev = std::sqrt((vec - vec.mean()).square().sum()/(vec.size()-1));
        cout << "mean " << vec.mean() << endl;
        cout << "stdv " << std_dev << endl;

        return 0;


        extract_shi_tomasi_corners(img_shi_tomasi, corners_shi_tomasi, sts);
        std::vector<cv::KeyPoint> keypoint_shi_tomasi;
        for(int i = 0; i < corners_shi_tomasi.size(); i++)
        {
            keypoint_shi_tomasi.push_back(cv::KeyPoint(corners_shi_tomasi[i].x, corners_shi_tomasi[i].y, 0 ) );
            img_shi_tomasi.at<cv::Vec3b>(cv::Point(corners_shi_tomasi[i].x, corners_shi_tomasi[i].y) ) = cv::Vec3b(rng_st.uniform(100, 255), rng_st.uniform(100, 255),rng_st.uniform(100, 255));
        }
        kps = keypoint_shi_tomasi;
        if(save)
        {
            imwrite("../images_features/feature_shi_tomasi_" + std::to_string(r) + "_" + std::to_string(beta_) + "_.png", img_shi_tomasi);
        }
        return corners_shi_tomasi.size();
    }
    else if(tof == TYPE_OF_FEATURES::SURF)
    {
        //Surf ///
        cv::Mat img_surf = img_.clone();
        std::vector<cv::KeyPoint> keypoint_surf;
        cv::Mat img_keypoints_surf;
        auto t1 = std::chrono::high_resolution_clock::now();
        Eigen::ArrayXd vec = ArrayXd::Zero(nb_times, 1);
        //cout << vec << endl << endl;
        for(int i = 0 ; i < nb_times; i++)
        {
            std::vector<cv::KeyPoint> keypoint_surf;
            auto start = std::chrono::high_resolution_clock::now();
            extract_surf_keypoints(img_surf, keypoint_surf, false);
            auto end = std::chrono::high_resolution_clock::now();
            vec(i) = duration_cast<microseconds>(end - start).count();
        }
        //cout << vec << endl << endl;

        auto t2 = std::chrono::high_resolution_clock::now();

        auto duration = duration_cast<microseconds>(t2 - t1).count();

        cout << "time took by " << duration/(1.0*nb_times) << " milliseconds" << endl;

        double std_dev = std::sqrt((vec - vec.mean()).square().sum()/(vec.size()-1));
        cout << "mean " << vec.mean() << endl;
        cout << "stdv " << calculateSD(vec) << endl;
        cout << "stdv " << std_dev << endl;

        return 0;

        extract_surf_keypoints(img_surf, keypoint_surf, false);
        cvtColor(img_surf, img_surf, cv::COLOR_GRAY2BGR);
        img_keypoints_surf = img_surf.clone();

        int r = 1;
        for (int i = 0; i < keypoint_surf.size(); i++)
        {
            img_keypoints_surf.at<cv::Vec3b>(keypoint_surf[i].pt) = cv::Vec3b(rng_st.uniform(100, 255), rng_st.uniform(100, 255),rng_st.uniform(100, 255));
        }
        kps = keypoint_surf;
        if(save)
        {
            imwrite("../images_features/feature_surf_" + std::to_string(r) + "_" + std::to_string(beta_) + "_.png", img_keypoints_surf);
        }
        return keypoint_surf.size();
    }
    else if(tof == TYPE_OF_FEATURES::FAST)
    {
        //Fast ///
        cv::Mat img_fast = img_.clone();
        std::vector<cv::KeyPoint> keypoint_fast;
        cv::Mat img_keypoints_fast;
        extract_fast_keypoints(img_fast, keypoint_fast);
        img_keypoints_fast = img_fast.clone();
        cvtColor(img_fast, img_keypoints_fast, cv::COLOR_GRAY2BGR);
        int r = 1;
        for (int i = 0; i < keypoint_fast.size(); i++)
        {
            img_keypoints_fast.at<cv::Vec3b>(keypoint_fast[i].pt) = cv::Vec3b(rng_st.uniform(100, 255), rng_st.uniform(100, 255),rng_st.uniform(100, 255));
        }
        kps = keypoint_fast;
        if(save)
        {
            imwrite("../images_features/feature_fast_" + std::to_string(r) + "_" + std::to_string(beta_) + "_.png", img_keypoints_fast);
        }
        return keypoint_fast.size();
    }
    else if(tof == TYPE_OF_FEATURES::BRISK)
    {

        //Brisk ///
        cv::Mat img_surf = img_.clone();
        std::vector<cv::KeyPoint> keypoint_brisk;
        cv::Mat img_keypoints_brisk;

        extract_brisk_features(img_surf, keypoint_brisk);
        cvtColor(img_surf, img_surf, cv::COLOR_GRAY2BGR);
        img_keypoints_brisk = img_surf.clone();

        for (int i = 0; i < keypoint_brisk.size(); i++)
        {
            //circle(img_keypoints_fast, keypoint_fast[i].pt, r, Scalar(rng_st.uniform(100, 255), rng_st.uniform(100, 255),rng_st.uniform(100, 255)), -1, -1, 0);
            img_keypoints_brisk.at<cv::Vec3b>(keypoint_brisk[i].pt) = cv::Vec3b(rng_st.uniform(100, 255), rng_st.uniform(100, 255),rng_st.uniform(100, 255));
        }
        kps = keypoint_brisk;
        if(save)
        {
            imwrite("../images_features/feature_brisk_" + std::to_string(r) + "_" + std::to_string(beta_) + "_.png", img_keypoints_brisk);
        }
        return keypoint_brisk.size();

    }
    else if(tof == TYPE_OF_FEATURES::AKAZE)
    {
        //akaze ///
        cv::Mat img_surf = img_.clone();
        std::vector<cv::KeyPoint> keypoint_akaze;
        cv::Mat img_keypoints_akaze;

        extract_akaze_features(img_surf, keypoint_akaze);
        cvtColor(img_surf, img_surf, cv::COLOR_GRAY2BGR);
        img_keypoints_akaze = img_surf.clone();

        for (int i = 0; i < keypoint_akaze.size(); i++)
        {
            img_keypoints_akaze.at<cv::Vec3b>(keypoint_akaze[i].pt) = cv::Vec3b(rng_st.uniform(100, 255), rng_st.uniform(100, 255),rng_st.uniform(100, 255));
        }
        kps = keypoint_akaze;
        if(save)
        {
            imwrite("../images_features/feature_akaze_" + std::to_string(r) + "_" + std::to_string(beta_) + "_.png", img_keypoints_akaze);
        }
        return keypoint_akaze.size();
    }
    else if(tof == TYPE_OF_FEATURES::GFTT)
    {
        //gftt ///
        cv::Mat img_gftt = img_.clone();
        std::vector<cv::KeyPoint> keypoint_gfft;
        cv::Mat img_keypoints_gftt;

        extract_gftt_features(img_gftt, keypoint_gfft);
        cvtColor(img_gftt, img_gftt, cv::COLOR_GRAY2BGR);
        img_keypoints_gftt = img_gftt.clone();

        for (int i = 0; i < keypoint_gfft.size(); i++)
        {
            //circle(img_keypoints_fast, keypoint_fast[i].pt, r, Scalar(rng_st.uniform(100, 255), rng_st.uniform(100, 255),rng_st.uniform(100, 255)), -1, -1, 0);
            img_keypoints_gftt.at<cv::Vec3b>(keypoint_gfft[i].pt) = cv::Vec3b(rng_st.uniform(100, 255), rng_st.uniform(100, 255),rng_st.uniform(100, 255));
        }
        kps = keypoint_gfft;
        if(save)
        {
            imwrite("../images_features/feature_gftt_" + std::to_string(r) + "_" + std::to_string(beta_) + "_.png", img_keypoints_gftt);
        }
        return keypoint_gfft.size();
    }
    else if(tof == TYPE_OF_FEATURES::ORB)
    {
        //ORB ///
        cv::Mat img_orb = img_.clone();
        std::vector<cv::KeyPoint> keypoint_orb;
        cv::Mat img_keypoints_orb;

        extract_orb_features(img_orb, keypoint_orb);
        cvtColor(img_orb, img_orb, cv::COLOR_GRAY2BGR);
        img_keypoints_orb = img_orb.clone();

        for (int i = 0; i < keypoint_orb.size(); i++)
        {
            img_keypoints_orb.at<cv::Vec3b>(keypoint_orb[i].pt) = cv::Vec3b(rng_st.uniform(100, 255), rng_st.uniform(100, 255),rng_st.uniform(100, 255));
        }
        kps = keypoint_orb;
        if(save)
        {
            imwrite("../images_features/feature_orb_" + std::to_string(r) + "_" + std::to_string(beta_) + "_.png", img_keypoints_orb);
        }
        return keypoint_orb.size();
    }
    else if(tof == TYPE_OF_FEATURES::AGAST)
    {
        //AGAST ///
        cv::Mat img_agast = img_.clone();
        std::vector<cv::KeyPoint> keypoint_agast;
        cv::Mat img_keypoints_agast;

        extract_agast_features(img_agast, keypoint_agast);
        cvtColor(img_agast, img_agast, cv::COLOR_GRAY2BGR);
        img_keypoints_agast = img_agast.clone();

        for (int i = 0; i < keypoint_agast.size(); i++)
        {
            img_keypoints_agast.at<cv::Vec3b>(keypoint_agast[i].pt) = cv::Vec3b(rng_st.uniform(100, 255), rng_st.uniform(100, 255),rng_st.uniform(100, 255));
        }
        kps = keypoint_agast;
        if(save)
        {
            imwrite("../images_features/feature_agast_" + std::to_string(r) + "_" + std::to_string(beta_) + "_.png", img_keypoints_agast);
        }
        return keypoint_agast.size();
    }
    else if(tof == TYPE_OF_FEATURES::PHASE_CONGRUENCY)
    {
    cv::Mat img_ph = img_.clone();


        std::vector<cv::KeyPoint> keypoint_ph;

        phase_congruency<float> pc_settings;
        phase_congruency_output_eigen pco(img_ph.rows, img_ph.cols, pc_settings);

        //phase_congruency_3(input_left_image, pco);


        return keypoint_ph.size();
    }

    return -1;
}

template <typename E>
constexpr auto to_underlying(E e) noexcept
{
    return static_cast<std::underlying_type_t<E>>(e);
}

int correct_detected(std::vector<cv::KeyPoint> first_keypoint, std::vector<cv::KeyPoint> second_keypoint)
{
    int corrected_detected = 0;
    for(size_t i = 0; i < first_keypoint.size(); i++)
    {
        for(size_t j = 0; j < second_keypoint.size(); j++)
        {
            if(int(first_keypoint[i].pt.x) == int(second_keypoint[j].pt.x)
                    && int(first_keypoint[i].pt.y) == int(second_keypoint[j].pt.y) )
            {
                corrected_detected++;
                break;
            }
        }
    }
    return corrected_detected;
}



void feature_extraction_liste(string experience_name, string sub_experience_name)
{

    std::vector<string> imageListCam1, imageListCam2;
    string inputFilename;


    string path_to_camera1_images = string("..//").append(experience_name).append("//").append(experience_name).append(sub_experience_name).append(string("//Camera1"));
    string path_to_camera2_images = string("..//").append(experience_name).append("//").append(experience_name).append(sub_experience_name).append(string("//Camera2"));

    imageListCam1 = get_list_of_files(path_to_camera1_images);
    imageListCam2 = get_list_of_files(path_to_camera2_images);


    size_t nimages = imageListCam1.size();



    for(int t = 0; t < 7; t++)
    {
        cout << "type of feature " << get_string_from_tof_enum(get_type_of_features_from_int(t)) << endl;
        TYPE_OF_FEATURES tof = get_type_of_features_from_int(t);

        /*string label_path = name_from_type_of_features(tof) + "_nb";
            create_directory(label_path);
            label_path = label_path + "//" + experience_name;
            create_directory(label_path);*/

        string dir_name("..//Number_of_features//");
        create_directory(dir_name);
        string path_to_all = string(dir_name).append("//").append(experience_name);
        create_directory(path_to_all);

        int scale_factor = 1;


        std::vector<int> list_nb_redectect(100, 0);
        std::vector<int> list_nb_feat(100, 0);

        //cout << "beta = " << beta_ << endl;
        for(size_t index_img = 0 ;index_img < 0 ; index_img++)
        {
            cout << "beta = " << index_img << endl;
            std::vector<cv::KeyPoint> first_keypoint;
            for(int beta_ = 0; beta_ < 100; beta_ ++)
            {


                cv::Mat left_view_16 = cv::imread(imageListCam1[index_img], CV_16U);
                cv::Mat left_view;

                double minVal, maxVal;
                cv::Point minLoc, maxLoc;
                cv::minMaxLoc(left_view_16, &minVal, &maxVal, &minLoc, &maxLoc);
                left_view_16.convertTo(left_view, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));

                cv::cvtColor(left_view, left_view, cv::COLOR_GRAY2BGR);

                left_view = change_brightness_image(left_view, 1,  beta_);

                if( scale_factor > 1)
                    cv::resize(left_view, left_view, cv::Size(), scale_factor, scale_factor , cv::INTER_CUBIC );

                cv::cvtColor(left_view, left_view, cv::COLOR_BGR2GRAY);

                std::vector<cv::KeyPoint> keypoint_feat;

                bool save = false;

                if(beta_ == 0 || beta_ == 50 || beta_ == 99)
                {
                    if(index_img % 100 == 0)
                        save = true;
                }


                int nb_feat = feature_extractor_main(left_view,  tof , keypoint_feat, index_img ,  beta_, save);

                int nb = 0;

                if(beta_ == 0)
                {
                    first_keypoint = keypoint_feat;
                }


                nb = correct_detected(first_keypoint, keypoint_feat);


                list_nb_feat[beta_] += nb_feat > 0 ? nb_feat : 0 ;

                list_nb_redectect[beta_] +=  nb;

            }
        }

        std::string all_temp_gc = path_to_all;
        all_temp_gc = all_temp_gc + "//all_number_features.csv";
        ofstream all_log_gc (all_temp_gc,  std::ios_base::app);
        for(size_t i = 0 ; i < list_nb_feat.size(); i++)
        {
            all_log_gc << list_nb_feat[i] <<";" ;
            if(i == list_nb_feat.size() - 1)
            {
                all_log_gc << list_nb_feat[i] << endl ;
            }
        }


        std::string path_to_detect = path_to_all;
        path_to_detect = path_to_detect + "//redetected.csv";
        ofstream log_detect (path_to_detect,  std::ios_base::app);
        for(size_t i = 0 ; i < list_nb_redectect.size(); i++)
        {
            log_detect << list_nb_redectect[i] <<";" ;
            if(i == list_nb_redectect.size() - 1)
            {
                log_detect << list_nb_redectect[i] << endl ;
            }
        }


    }
}

void compute_feature_extraction_execution_time(string experience_name, string sub_experience_name)
{

    std::vector<string> imageListCam1, imageListCam2;
    string inputFilename;


    string path_to_camera1_images = string("..//").append(experience_name).append("//").append(experience_name).append(sub_experience_name).append(string("//Camera1"));
    string path_to_camera2_images = string("..//").append(experience_name).append("//").append(experience_name).append(sub_experience_name).append(string("//Camera2"));

    imageListCam1 = get_list_of_files(path_to_camera1_images);
    imageListCam2 = get_list_of_files(path_to_camera2_images);


    size_t nimages = imageListCam1.size();



    for(int t = 0; t < 7; t++)
    {
        cout << "type of feature " << get_string_from_tof_enum(get_type_of_features_from_int(t)) << endl;
        TYPE_OF_FEATURES tof = get_type_of_features_from_int(t);

        /*string label_path = name_from_type_of_features(tof) + "_nb";
            create_directory(label_path);
            label_path = label_path + "//" + experience_name;
            create_directory(label_path);*/

        string dir_name("..//Number_of_features//");
        create_directory(dir_name);
        string path_to_all = string(dir_name).append("//").append(experience_name);
        create_directory(path_to_all);

        int scale_factor = 1;


        std::vector<int> list_nb_redectect(100, 0);
        std::vector<int> list_nb_feat(100, 0);

        //cout << "beta = " << beta_ << endl;
        for(size_t index_img = 0 ;index_img < 0 ; index_img++)
        {
            cout << "beta = " << index_img << endl;
            std::vector<cv::KeyPoint> first_keypoint;
            for(int beta_ = 0; beta_ < 100; beta_ ++)
            {


                cv::Mat left_view_16 = cv::imread(imageListCam1[index_img], CV_16U);
                cv::Mat left_view;

                double minVal, maxVal;
                cv::Point minLoc, maxLoc;
                cv::minMaxLoc(left_view_16, &minVal, &maxVal, &minLoc, &maxLoc);
                left_view_16.convertTo(left_view, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));

                cvtColor(left_view, left_view, cv::COLOR_GRAY2BGR);

                left_view = change_brightness_image(left_view, 1,  beta_);

                if( scale_factor > 1)
                    cv::resize(left_view, left_view, cv::Size(), scale_factor, scale_factor , cv::INTER_CUBIC );

                cvtColor(left_view, left_view, cv::COLOR_BGR2GRAY);

                std::vector<cv::KeyPoint> keypoint_feat;

                bool save = false;

                if(beta_ == 0 || beta_ == 50 || beta_ == 99)
                {
                    if(index_img % 100 == 0)
                        save = true;
                }


                int nb_feat = feature_extractor_main(left_view,  tof , keypoint_feat, index_img ,  beta_, save);

                int nb = 0;

                if(beta_ == 0)
                {
                    first_keypoint = keypoint_feat;
                }


                nb = correct_detected(first_keypoint, keypoint_feat);


                list_nb_feat[beta_] += nb_feat > 0 ? nb_feat : 0 ;

                list_nb_redectect[beta_] +=  nb;

            }
        }

        std::string all_temp_gc = path_to_all;
        all_temp_gc = all_temp_gc + "//all_number_features.csv";
        ofstream all_log_gc (all_temp_gc,  std::ios_base::app);
        for(size_t i = 0 ; i < list_nb_feat.size(); i++)
        {
            all_log_gc << list_nb_feat[i] <<";" ;
            if(i == list_nb_feat.size() - 1)
            {
                all_log_gc << list_nb_feat[i] << endl ;
            }
        }


        std::string path_to_detect = path_to_all;
        path_to_detect = path_to_detect + "//redetected.csv";
        ofstream log_detect (path_to_detect,  std::ios_base::app);
        for(size_t i = 0 ; i < list_nb_redectect.size(); i++)
        {
            log_detect << list_nb_redectect[i] <<";" ;
            if(i == list_nb_redectect.size() - 1)
            {
                log_detect << list_nb_redectect[i] << endl ;
            }
        }


    }
}


}
