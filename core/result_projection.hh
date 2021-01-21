#pragma once

#include <Eigen/Dense>
#include <Eigen/Core>
#include <iostream>
#include <memory>
#include "stereo_match/stereo_match.hh"


using namespace Eigen;
using namespace std;


namespace tpp
{
/*
    template <typename T>
    struct result_projection
    {
        result_projection()
        {

        }
        result_projection(Matrix<T, 2, 1> lp, Matrix<T, 2, 1> rp, Matrix<T, 3, 1> pp)
        {
            left_point = lp;
            right_point = rp;
            point_projected = pp;

        }
        Matrix<T, 2, 1> left_point;
        Matrix<T, 2, 1> right_point;
        Matrix<T, 3, 1> point_projected;
        std::vector< Matrix<T, 2, 1> > history_left;
        std::vector< Matrix<T, 2, 1> > history_right;
        int taken = -1;
    };/**/

    typedef std::vector<stereo_match>  projected_object;

    typedef std::vector< projected_object > projected_objects_image;

    typedef std::vector<Vector2i> markers;


    struct object_segmented
    {
        object_segmented()
        {

        }

        projected_object list_of_object_points;
        std::vector<float> disparity_in_x;
        std::vector<float> disparity_in_y;
        int age;
        int time_to_live = 5;
    };

    struct ascending_depth_reprojection
    {
        // 3d points are in the format z, y, x
        inline bool operator() (const stereo_match &point3d1, const stereo_match &point3d2)
        {
            return (point3d1.point_projected(0) < point3d2.point_projected(0));
        }
    };

    struct sort_mean_descending
    {
        inline bool operator() (const Vector4d& point3d1, const Vector4d& point3d2)
        {
            return (point3d1(1) < point3d2(1));
        }
    };



}


