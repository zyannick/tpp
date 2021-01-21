#pragma once


#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>

using namespace Eigen;
using namespace std;

struct bubble
{
    bubble() {}
    bubble(double disp_x, double disp_y);
    bubble(double disp_x, double disp_y, string file_name);
    double disparity_in_x;
    double disparity_in_y;
    string all_points_file_name;

};
