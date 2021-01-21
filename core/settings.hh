#pragma once

#include <iostream>
#include <string>
#include "algorithms/miscellaneous.hh"
#include "type_of_features.hh"
#include "type_of_matching.hh"
#include <experimental/filesystem>



using namespace std;
namespace tpp {


template< typename T>
struct base_options
{
    int wind_row;
    int wind_col;
    string experience_name;
    string sub_experience_name;
    double eps;
    int scale_factor;
    int wind_gravity_center;
    int surface_min_person;
    int time_to_live;
    bool rect;
    TYPE_OF_FEATURES tof = TYPE_OF_FEATURES::PHASE_CONGRUENCY;
    MATCHING_SIMILARITY_METHOD dist_method = MATCHING_SIMILARITY_METHOD::LADES_SIMILARITY;
    SUB_PIXEL_MATCHING_PRECISE sub_pixel = SUB_PIXEL_MATCHING_PRECISE::PHASE_CORRELATION_FORROSH;
    bool plane_z_mean = false;
    bool with_map_ref = false;
    bool del_outliers = false;
    bool verbose = true;
    bool motion;
    bool keep_values;
    bool ensur_unique = true;
    bool ensure_consistency = true;
    int type_extraction = 0;
    int phase_cong_type = 1;
    bool save_images ;
    string root_repository = "";
    std::vector<string> list_sub_dirs;
    bool bit16 = false;


    base_options ()
    {
        wind_row = 7;
        wind_col = 7;
        experience_name = "";
        sub_experience_name = "";
        scale_factor = 1;
        wind_gravity_center = 6;
        surface_min_person = 10;
        time_to_live = 5;
        rect = true;
        tof = TYPE_OF_FEATURES::ORB;
        dist_method = MATCHING_SIMILARITY_METHOD::LADES_SIMILARITY;
        sub_pixel = SUB_PIXEL_MATCHING_PRECISE::PHASE_CORRELATION_FORROSH;
        plane_z_mean = false;
        with_map_ref = false;
        del_outliers = false;
        verbose = true;
        motion = false;
        keep_values = true;
        save_images = true;
        bool bit16 = false;
    }

    base_options (string json_file)
    {
        namespace fs = std::experimental::filesystem;
        if(json_file.empty() || !fs::exists(json_file))
        {
            base_options();
        }
        else
        {
            cv::FileStorage fs2(json_file, cv::FileStorage::READ);
            fs2["wind_row"] >> wind_row;
            fs2["wind_col"] >> wind_col;
            fs2["scale_factor"] >> scale_factor;
            fs2["wind_gravity_center"] >> wind_gravity_center;
            fs2["surface_min_person"] >> surface_min_person;
            fs2["time_to_live"] >> time_to_live;
            fs2["rect"] >> rect;
            fs2["plane_z_mean"] >> plane_z_mean;
            fs2["with_map_ref"] >> with_map_ref;
            fs2["del_outliers"] >> del_outliers;
            fs2["verbose"] >> verbose;
            fs2["motion"] >> motion;
            fs2["keep_values"] >> keep_values;
            fs2["save_images"] >> save_images;
            fs2["root_repository"] >> root_repository;
            fs2["bit16"] >> bit16;

            cout << "here testing " << keep_values << endl;

            list_sub_dirs = get_list_of_files("..//" +  root_repository);
            for (int i = 0; i < list_sub_dirs.size(); i++ )
            {
                std::vector<string> var_list = split(list_sub_dirs[i], "//");
                //cout << var_list[var_list.size()-1] << endl;
                list_sub_dirs[i] = var_list[var_list.size()-1];
            }
        }
    }
};

template< typename T>
struct phase_congruency_options
{
    int wind_row;
    int wind_col;
    string experience_name;
    string sub_experience_name;
    double eps;
    int scale_factor;
    int wind_gravity_center;
    int surface_min_person;
    int time_to_live;
    bool rect;
    TYPE_OF_FEATURES tof;
    MATCHING_SIMILARITY_METHOD dist_method;
    SUB_PIXEL_MATCHING_PRECISE sub_pixel;
    bool plane_z_mean = false;
    bool with_map_ref = false;
    bool del_outliers = false;
    bool verbose = true;
    bool motion;
    bool keep_values;
    bool ensur_unique = true;
    bool ensure_consistency = true;

    phase_congruency_options ()
    {
        wind_row = 7;
        wind_col = 7;
        experience_name = "";
        sub_experience_name = "";
        scale_factor = 4;
        wind_gravity_center = 6;
        surface_min_person = 10;
        time_to_live = 5;
        rect = true;
        tof = TYPE_OF_FEATURES::PHASE_CONGRUENCY;
        dist_method = MATCHING_SIMILARITY_METHOD::LADES_SIMILARITY;
        sub_pixel = SUB_PIXEL_MATCHING_PRECISE::PHASE_CORRELATION_FORROSH;
        plane_z_mean = false;
        with_map_ref = false;
        del_outliers = false;
        verbose = true;
        motion = true;
        keep_values = true;
    }
};

}
