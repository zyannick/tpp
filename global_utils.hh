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
#include <experimental/filesystem>

using namespace std;
using namespace Eigen;


namespace tpp {


bool create_directory(string dir_name, bool verbose = false)
{
    namespace fs = std::experimental::filesystem;
    string string_dir_name(dir_name);
    if(!fs::is_directory(string_dir_name) || !fs::exists(string_dir_name))
    {
        if(fs::create_directory(string_dir_name))
        {
            if(verbose)
            cout << "Directory " << string_dir_name << " created" << endl;
            return true;
        }
        else {
            if(verbose)
            cout << "Directory " << string_dir_name << " not created" << endl;
        }
    }
    else {
        if(verbose)
        cout <<  string_dir_name << " already exists or is not a directory" << endl;
    }

    return false;

}

void delete_before_create_directory(string dir_name)
{
    namespace fs = std::experimental::filesystem;
    string string_dir_name(dir_name);
    if(fs::is_directory(string_dir_name) && fs::exists(string_dir_name))
    {
        fs::remove_all(string_dir_name);
    }

    if(!fs::is_directory(string_dir_name) || !fs::exists(string_dir_name))
    {
        fs::create_directory(string_dir_name);
    }
}

}
