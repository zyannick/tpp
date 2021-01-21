#pragma once
#include "calibration/stereo_calibration.hh"
#include "recontruction3D/recontruction3D.hh"

namespace tpp
{

void calib_and_3d(Type_Of_Result type, Type_Of_Calibration type_calib, USE_RECTIFIED_3D_POINTS_GRID use_rect,
                  string file_name = "imageList.old.json", string file_objet = "file_object.json")
{

    cout << "calib_and_3d" << endl;
    if (Type_Of_Result::ONLY_CALIBRATION == type || Type_Of_Result::CALIBRATION_AND_3D_RECONSTRUCTION == type)
    {

        cv::Mat cameraMatrix[2], distCoeffs[2];
        std::vector<string> left_images;
        std::vector<string> right_images;
        size_t nimages;
        cv::Mat R;
        cv::Mat T;
        cv::Mat E;
        cv::Mat F;
        cv::Mat Q;
        string path_to_calibration_files = "TSRD_CALIBRATION";

        int wind_size = 6;

        //for(wind_size = 3 ; wind_size <= 15; wind_size++)
        stereo_calibration_routine_ransac(path_to_calibration_files,wind_size,cameraMatrix, distCoeffs,
                                   left_images, right_images,
                                   R, T, E, F, Q, DIRECTORY, type_calib, use_rect);
        /*stereo_calibration(cameraMatrix, distCoeffs, left_images, right_images,
                                                           R, T, E, F, Q, DIRECTORY, type_calib, use_rect);*/
    }
    else if (Type_Of_Result::_3D_RECONSTRUCTION == type)
    {

        cout << "_3D_RECONSTRUCTION" << endl;
        stereo_params_cv stereo_par;
        stereo_par.retreive_values();

        plane hori_plane;
        cv::Size imageSize;
        int nimages_plane = 1;
        std::vector<cv::Mat> list_3D_points_h_plane;
        std::vector<plane> list_plane;
        std::vector<std::vector<cv::Point2f>> list_points_plan_left;
        std::vector<std::vector<cv::Point2f>> list_points_plan_right;

        cout << "plane_reconstruction" << endl;
        int nimages = 0;
        bool is_ground = false;
        bool only_marked = false;
        plane_reconstruction_marked(Mode_To_Revtreive_Files::JSFILES, list_points_plan_left, list_points_plan_right,nimages,is_ground, only_marked);

        size_t sz  = list_points_plan_left[0].size();

        if(sz == 1)
        {

            std::vector<cv::Point2f> points_plan_left;
            std::vector<cv::Point2f> points_plan_right;
            cv::Mat _3D_plan;
            plane pl;

            cout << "size " << list_points_plan_left.size() << endl;

            for(size_t i = 0 ; i < list_points_plan_left.size() ;  i++ )
            {
                //cout << "left " << list_points_plan_left[i][0] << " right " << list_points_plan_right[i][0] << endl;
                points_plan_left.push_back(list_points_plan_left[i][0]);
                points_plan_right.push_back(list_points_plan_right[i][0]);
            }

            cout << "on y est " << endl;

            points_3D_reconstruction_rectified(stereo_par, points_plan_left, points_plan_right, _3D_plan, pl, FIT_PLANE_CLOUD_POINT::FIT);

            save_3D_points(_3D_plan,9);
            ofstream myP;
            myP.open("plane.txt");
            myP << " a = " << (pl).A << ";" << " b = " << (pl).B << ";" << " c = " << (pl).C << ";" << " d = " << (pl).D << "; \n";
            myP.close();

            cout << "Distance between points and the regression plane(mm) :" << endl;
            distance_3D_point_to_plane(pl.A, pl.B, pl.D, _3D_plan);

            plane ground_plane(0.86608,-0.881973,-1,1924.86);

            if(is_ground)
            {
                ground_plane = pl;
            }
            else
            {
                cout << endl << "Distance between points and the regression plane of the ground(mm) :" << endl;
                distance_3D_point_to_plane(ground_plane.A, ground_plane.B, ground_plane.D, _3D_plan);
            }

/**/
        }
        else if (sz >= 3)
        {
            cout << "points_3D_reconstruction_rectified " << endl;
            points_3D_reconstruction_list_rectified(stereo_par, list_points_plan_left, list_points_plan_right, nimages_plane, imageSize, list_3D_points_h_plane, list_plane, FIT_PLANE_CLOUD_POINT::FIT);
            save_3D_points(list_3D_points_h_plane);
            assert(list_3D_points_h_plane.size() == 1 && "Only one plane is need");
        }

        //system("PAUSE");
    }
}/**/
}
