#pragma once

#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"

#include "opencv2/highgui.hpp"
#include <vector>
#include <iostream>
#include "algorithms/feature_extractor.hh"
#include "core.hpp"

using namespace std;
 

namespace tpp {

void feature_extraction_selector(Mat img, std::vector<KeyPoint> &keypoints,Mat &descriptors, TYPE_OF_FEATURES tof)
{
    if(TYPE_OF_FEATURES::ORB == tof)
    {
        extract_orb_features_descriptors(img, keypoints, descriptors);
    }
    else if(TYPE_OF_FEATURES::FAST == tof)
    {
        extract_fast_keypoints_descriptors(img, keypoints, descriptors);
    }
    else if(TYPE_OF_FEATURES::GFTT == tof)
    {
        extract_gftt_features_descriptors(img, keypoints, descriptors);
    }
    else if(TYPE_OF_FEATURES::KAZE == tof)
    {
        extract_kaze_features_descriptors(img, keypoints, descriptors);
    }
    else if(TYPE_OF_FEATURES::SURF == tof)
    {
        extract_surf_keypoints_descriptors(img, keypoints, descriptors);
    }
    else if(TYPE_OF_FEATURES::AGAST == tof)
    {
        extract_agast_features_descriptors(img, keypoints, descriptors);
    }
    else if(TYPE_OF_FEATURES::BRISK == tof)
    {
        extract_brisk_features_descriptors(img, keypoints, descriptors);
    }
    else if(TYPE_OF_FEATURES::SHI_TOMASI == tof)
    {
        extract_fast_keypoints_descriptors(img, keypoints, descriptors);
    }
}


std::vector<stereo_match> from_keypoints_to_stereo_matches(std::vector< KeyPoint >  	keypoints1, std::vector< KeyPoint >  	keypoints2,
                                                           std::vector< DMatch >  	matches1to2 )
{
    std::vector<stereo_match> results;
    
    for(size_t i = 0 ; i < matches1to2.size(); i++ )
    {
        int oricy = matches1to2[i].queryIdx;
        int dest = matches1to2[i].trainIdx;
        Vector2d left_point;
        Vector2d right_point;

        left_point.x() = keypoints1[oricy].pt.x;
        left_point.y() = keypoints1[oricy].pt.y;

        right_point.x() = keypoints2[dest].pt.x;
        right_point.y() = keypoints2[dest].pt.y;


        results.push_back(stereo_match(left_point, right_point, 0) );
    }
    return results;

}

void cv_stereo_match(Mat left_img, Mat right_img, std::vector<stereo_match> &list_matches,
                     TYPE_OF_FEATURES tof, bool verbose, int index_img = 0)
{
    std::vector<KeyPoint> left_keypoints, right_keypoints;
    Mat left_descriptors, right_descriptors;

    feature_extraction_selector(left_img, left_keypoints, left_descriptors, tof);

    Mat out_img;

    cv::drawKeypoints(left_img, left_keypoints, out_img);
    cv::imwrite("features" + std::to_string(index_img) +  ".png", out_img);

    feature_extraction_selector(right_img, right_keypoints, right_descriptors, tof);

    //
    if(verbose)
    cout << "left keypoints " << left_keypoints.size() << "  right keypoints " << right_keypoints.size() << endl;

    if( left_keypoints.size() <= 1 || right_keypoints.size() <= 1 )
    {
        if(verbose)
        cout << "Pas de features " << endl;
        return;
    }


    if(left_descriptors.type()!=CV_32F) {
        left_descriptors.convertTo(left_descriptors, CV_32F);
    }

    if(right_descriptors.type()!=CV_32F) {
        right_descriptors.convertTo(right_descriptors, CV_32F);
    }

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<DMatch> > knn_matches;
    matcher->knnMatch( left_descriptors, right_descriptors, knn_matches, 2 );

    //-- Step 3: Matching descriptor vectors using FLANN matcher
    /*FlannBasedMatcher matcher;
    std::vector< DMatch > knn_matches;
    matcher.match( left_descriptors, right_descriptors, knn_matches );*/
    if(verbose)
    cout << "taille de matches" << knn_matches.size() << endl;

    if( knn_matches.size() < 1)
        return;

    //-- Quick calculation of max and min distances between keypoints
    const float ratio_thresh = 0.7f;
    std::vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    
    //cout << "taille de good match " << good_matches.size() << endl;
    
    list_matches = from_keypoints_to_stereo_matches(left_keypoints, right_keypoints, good_matches);
    if(verbose)
    cout << "taille stereo_matches  " << list_matches.size() << endl;
}


}
