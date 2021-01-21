#pragma once

#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"

 
using namespace cv::xfeatures2d;

namespace tpp
{
	void extract_fast_keypoints(cv::Mat img_1, std::vector<cv::KeyPoint> &keypoints_1)
	{
		//-- Step 1: Detect the keypoints using SURF Detector
        int min_hessian = 300;
        cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
		detector->detect(img_1, keypoints_1, cv::Mat());
		return;
	}

    void extract_surf_keypoints(cv::Mat img_1, std::vector<cv::KeyPoint> &keypoints_1, bool up_r =false)
	{
		//-- Step 1: Detect the keypoints using SURF Detector
        int min_hessian = 100;
        int max_hessian = 500;
        //cv::Ptr<cv::SURF> detector = cv::SURF::create(min_hessian);
		//detector->setUpright(up_r);
		//detector->detect(img_1, keypoints_1);
		return;
	}

	void extract_brisk_features(cv::Mat image, std::vector<cv::KeyPoint> &briskKeypoints)
	{
        cv::Ptr<cv::BRISK> briskDetector = cv::BRISK::create(1);
		briskDetector->detect(image, briskKeypoints);
		return;
	}

	void extract_akaze_features(cv::Mat image, std::vector<cv::KeyPoint> &akazeKeypoints)
	{
		cv::Ptr<cv::AKAZE> akazeDetector = cv::AKAZE::create();
		akazeDetector->detect(image, akazeKeypoints);
		return;
	}

	void extract_agast_features(cv::Mat image, std::vector<cv::KeyPoint> &agastKeypoints)
	{
        cv::Ptr<cv::AgastFeatureDetector> agastDetector = cv::AgastFeatureDetector::create();
		agastDetector->detect(image, agastKeypoints);
		return;
	}

	void extract_gftt_features(cv::Mat image, std::vector<cv::KeyPoint> &gfttKeypoints)
	{
        cv::Ptr<cv::GFTTDetector> gfttDetector = cv::GFTTDetector::create(1000, 0.01, 1,
                                                                          3, false, 0.04);
		gfttDetector->detect(image, gfttKeypoints);
		return;
	}

	void extract_kaze_features(cv::Mat image, std::vector<cv::KeyPoint> &kazeKeypoints)
	{
		cv::Ptr<cv::KAZE> kazeDetector = cv::KAZE::create();
		kazeDetector->detect(image, kazeKeypoints);
		return;
	}

	void extract_orb_features(cv::Mat image, std::vector<cv::KeyPoint> &orbKeypoints)
	{
        cv::Ptr<cv::ORB> orbDetector = cv::ORB::create(5000, 1.2f, 8, 3,
                                                       0, 2, cv::ORB::HARRIS_SCORE, 3, 20);
		orbDetector->detect(image, orbKeypoints);
		return;
	}


    void extract_mser_features(cv::Mat image, std::vector<cv::KeyPoint> &keypoints_)
    {
        cv::Ptr<cv::MSER> ms = cv::MSER::create();
        std::vector<std::vector<cv::Point> > regions;
        std::vector<cv::Rect> mser_bbox;
        ms->detectRegions(image, regions, mser_bbox);
        //std::cout << "val " << regions.size() << "  "
        return;
    }

    void extract_fast_keypoints_descriptors(cv::Mat img_1, std::vector<cv::KeyPoint> &fast_keypoints, cv::Mat &descriptors)
    {
        //-- Step 1: Detect the keypoints using SURF Detector
        int min_hessian = 300;
        cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
        detector->detectAndCompute(img_1, cv::Mat(), fast_keypoints, descriptors);
        return;
    }

    void extract_surf_keypoints_descriptors(cv::Mat img_1, std::vector<cv::KeyPoint> &surf_keypoints, cv::Mat &descriptors, bool up_r =false)
    {
        //-- Step 1: Detect the keypoints using SURF Detector
        int min_hessian = 1000;
        int max_hessian = 500;
        //cv::Ptr<cv::SURF> detector = cv::SURF::create(300, 16, 9);
        //detector->setUpright(up_r);
        //detector->detect(img_1, surf_keypoints);
        //detector->detectAndCompute(img_1, cv::Mat(), surf_keypoints, descriptors);
        return;
    }

    void extract_brisk_features_descriptors(cv::Mat image, std::vector<cv::KeyPoint> &briskKeypoints, cv::Mat &descriptors)
    {
        cv::Ptr<cv::BRISK> briskDetector = cv::BRISK::create();
        briskDetector->detectAndCompute(image, cv::Mat(), briskKeypoints, descriptors);
        return;
    }

    void extract_akaze_features_descriptors(cv::Mat image, std::vector<cv::KeyPoint> &akazeKeypoints, cv::Mat &descriptors)
    {
        cv::Ptr<cv::AKAZE> akazeDetector = cv::AKAZE::create();
        akazeDetector->detectAndCompute(image, cv::Mat(), akazeKeypoints, descriptors);
        return;
    }

    void extract_agast_features_descriptors(cv::Mat image, std::vector<cv::KeyPoint> &agastKeypoints, cv::Mat &descriptors)
    {
        cv::Ptr<cv::AgastFeatureDetector> agastDetector = cv::AgastFeatureDetector::create();
        agastDetector->detectAndCompute(image, cv::Mat(), agastKeypoints, descriptors);
        return;
    }

    void extract_gftt_features_descriptors(cv::Mat image, std::vector<cv::KeyPoint> &gfttKeypoints, cv::Mat &descriptors)
    {
        cv::Ptr<cv::GFTTDetector> gfttDetector = cv::GFTTDetector::create();
        gfttDetector->detectAndCompute(image, cv::Mat(), gfttKeypoints, descriptors);
        return;
    }

    void extract_kaze_features_descriptors(cv::Mat image, std::vector<cv::KeyPoint> &kazeKeypoints, cv::Mat &descriptors)
    {


        cv::Ptr<cv::KAZE> kazeDetector = cv::KAZE::create();
        kazeDetector->detectAndCompute(image, cv::Mat(), kazeKeypoints, descriptors);
        return;
    }

    void extract_orb_features_descriptors(cv::Mat image, std::vector<cv::KeyPoint> &orbKeypoints, cv::Mat &descriptors)
    {
        cv::Ptr<cv::ORB> orbDetector = cv::ORB::create(5000, 1.2f, 8, 3,
                                                       0, 2, cv::ORB::HARRIS_SCORE, 3, 20);
        orbDetector->detectAndCompute(image, cv::Mat(),  orbKeypoints, descriptors);
        return;
    }



}
