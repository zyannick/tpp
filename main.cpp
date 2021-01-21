// tpp.cpp : définit le point d'entrée de l'application.
//

#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING

#include <iostream>
#include "tpp.h"

using namespace std;
using namespace Eigen;
using namespace tpp;

#ifdef VPP
#include <vpp/vpp.hh>
#include <vpp/utils/opencv_bridge.hh>
#include <vpp/utils/opencv_utils.hh>
using namespace vpp;
#endif



int parse_arg(int nb, char** argv)
{
    if (nb < 2)
    {
        return -1;
    }
    else if (nb == 2)
    {
        std::string str(argv[0]);
        if (str.compare("help"))
        {
            cout << "------------------------------------------------------------------" << endl <<
                "Thermal ++ " << endl
                << "------------------------------------------------------------------" << endl
                << "n: nothing" << endl
                << "fl: feature labelling" << endl
                << "fe: feature extraction" << endl
                << "s: stereo" << endl
                << "dp: disparity map" << endl;
        }
    }

    for (int i = 0; i < nb; i++)
    {

    }
}

void make_images_visibles_same_dir()
{
    cout << "make_images_visibles_same_dir" << endl;
    string root_dir = "./lab";
    std::vector<string> sub_dirs;
    sub_dirs.push_back("lab_green");
    sub_dirs.push_back("lab_orange");
    sub_dirs.push_back("lab_red");

    string new_root = "./together_lab";
    std::experimental::filesystem::remove_all(new_root);
    tpp::create_directory(new_root);
    int index_img = 1;
    for (int i = 0; i < sub_dirs.size(); i++)
    {
        std::vector<string> list_files = get_list_of_files(root_dir + "//" + sub_dirs[i]);

        cout << "sub_dir " << sub_dirs[i] << endl;

        for (int j = 0; j < list_files.size(); j++)
        {

            cout << "image " << j << endl;

            Mat img_16 = imread(list_files[j], CV_16U);
            Mat img;
            double minVal, maxVal;
            Point minLoc, maxLoc;
            minMaxLoc(img_16, &minVal, &maxVal, &minLoc, &maxLoc);
            img_16.convertTo(img, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
            imwrite(new_root + "//" + sub_dirs[i] + "_" + std::to_string(1000000 + index_img) + ".png", img);
            index_img++;
        }
    }
}

/*Mat img_cv = imread("img.png", 0);

//image2d<uchar> img = from_opencv<uchar>(img_cv);

image2d<float> img(5,6);
int nrows = img.nrows();
int ncols = img.ncols();
pixel_wise(img, img.domain()) | [&] (auto &i, auto coord)
{
    int row = coord[0];
    int col = coord[1];
    i = col * cos(col) + row * sin(row);
};

phase_congruency<float> pc_set;
phase_congruency_output_vpp<float> pc_output(img.nrows(), img.ncols(),
                                         pc_set);
pc_output.initialize_for_pc();
pc_output.initialize_moments();
phase_congruency_opt(img, pc_output);*/



int main(int argc, char** argv)
{

    int cl = parse_arg(argc, argv);

    MODULE_SELECTION selected_module;
    if (cl == -1)
    {
        selected_module = MODULE_SELECTION::DISPARITY_MAP;
    }
    else
    {

    }

    selected_module = MODULE_SELECTION::FEATURE_LABELLING;

    if (MODULE_SELECTION::DISPARITY_MAP == selected_module)
    {
        //std::string str = "100.png";
        //feature_extractor_main(str.c_str(),4);
        //disparity_map_liste(7,7);

        std::vector<string> list_experience;
        list_experience.push_back("Poseidon");
        list_experience.push_back("Thales");
        list_experience.push_back("Uliss");

        int wind_row = 7;
        int wind_col = 7;
        string sud_experience_name = "Ground";
        size_t nb_wall = 0;
        bool is_point = true;
        int scale_factor = 1;
        int wind_size_gravity_center_point_detection = 6;
        float eps = 0.01;

        for (size_t i = 0; i < list_experience.size(); i++)
        {
            string experience_name = list_experience[i];
            nb_wall = 0;
            cout << "No walls " << experience_name << endl;
            ground_segmentation(wind_row, wind_col, experience_name, nb_wall,
                is_point, eps, scale_factor,
                wind_size_gravity_center_point_detection);
            nb_wall = 1;
            cout << "One wall " << experience_name << endl;
            ground_segmentation(wind_row, wind_col, experience_name, nb_wall,
                is_point, eps, scale_factor,
                wind_size_gravity_center_point_detection);
            if (experience_name.compare("Poseidon") == 0
                || experience_name.compare("Thales") == 0)
            {
                nb_wall = 2;
                cout << "Two walls " << experience_name << endl;
                ground_segmentation(wind_row, wind_col, experience_name, nb_wall,
                    is_point, eps, scale_factor,
                    wind_size_gravity_center_point_detection);
            }
        }
        return 7;
    }
    else if (MODULE_SELECTION::FEATURE_LABELLING == selected_module)
    {

        POINT_OR_SILHOUETTE p_or_s = POINT_OR_SILHOUETTE::SILHOUETTE;


        if (POINT_OR_SILHOUETTE::POINT == p_or_s)
        {

            string experience_name = "Tuulmen";
            string sub_experience_name = "Sol";

            dense_ground_labelling(experience_name, sub_experience_name, 1, 3, true);

            sub_experience_name = "Haut";

            dense_ground_labelling(experience_name, sub_experience_name, 1, 3, false);

        }
        else if (POINT_OR_SILHOUETTE::SILHOUETTE == p_or_s)
        {

            base_options<float> stm("params.json");

            string experience_name = stm.root_repository;
            std::vector<string> sub_experience_name_list = stm.list_sub_dirs;


            for (int si = 0; si < sub_experience_name_list.size(); si++)
            {
                string sub_experience_name = sub_experience_name_list[si];


                stm.experience_name = experience_name;
                stm.sub_experience_name = sub_experience_name_list[si];
                for (int t = 0; t < 10; t++)
                {
                    //if(t==0 || t==1 || t==4)
                    if (t == 0)
                    {
                        if (stm.motion)
                        {
                            stm.eps = 0.1;
                            dense_disparity_map_liste_labelling_motion(stm);

                            /*stm.eps = 0.3;
                            dense_disparity_map_liste_labelling_motion(stm);

                            stm.eps = 0.01;
                            dense_disparity_map_liste_labelling_motion(stm);*/
                        }
                        else
                        {
                            cout << "feature extractor " << get_string_from_tof_enum(get_type_of_features_from_int(t)) << endl;
                            cout << "**********************************************************0.3**********************************************************" << endl;
                            stm.eps = 0.1;
                            cout << "keep values " << stm.keep_values << endl;
                            stm.experience_name = experience_name;
                            stm.sub_experience_name = sub_experience_name_list[si];
                            dense_disparity_map_liste_labelling(stm);
                        }


                    }
                }
            }

        }

        return 7;
    }
    else if (MODULE_SELECTION::STEREO == selected_module)
    {

        cout << "calib_and_3d" << endl;
        calib_and_3d(Type_Of_Result::CALIBRATION_AND_3D_RECONSTRUCTION,
            Type_Of_Calibration::STEREO, USE_RECTIFIED_3D_POINTS_GRID::YES);
        return 7;
    }
    else if (MODULE_SELECTION::FEATURE_EXTRACTION == selected_module)
    {

        std::vector<KeyPoint> kps;
        feature_extractor_main(imread("img.png", 0), TYPE_OF_FEATURES::SHI_TOMASI, kps);
        string experience_name = "Ultimate";
        string sub_experience_name = "Fall1";

        //test_disparity_map_liste(7,7);
        //feature_extraction_liste(experience_name, sub_experience_name);
        //feature_extractor_main(image_file_name.c_str(), 5);

        return 7;
    }
    else if (MODULE_SELECTION::OPTICAL_FLOW == selected_module)
    {
        open_cv_optical_flow("..//Tuulmen//TuulmenPerson1//Camera1");
        //generate_data_flow("../output_flow","..//Tuulmen//TuulmenPerson1//Camera1", "pref");
        generate_data_flow("train_all", "..//data_16//red_lab_16", "red");
        generate_data_flow("train_all", "..//data_16//green_lab_16", "green");
        generate_data_flow("train_all", "..//data_16//orange_lab_16", "orange");

        generate_data_flow("test_all", "..//data_16//orange_atrium_16", "orange");
        generate_data_flow("test_all", "..//data_16//red_atrium_16", "red");
        return 7;

    }
    else
    {
        cout << "It works but nothing is selected" << endl;
        return 7;
    }/**/
}
