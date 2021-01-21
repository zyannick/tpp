#pragma once

#include "algorithms/generate_maps/sparse_disparity_maps.hh"
#include "algorithms/segmentation.hh"
#include "algorithms/fitting.hh"
#include "global_utils.hh"
#include "algorithms/kernels.hh"
#include "core.hpp"


namespace tpp {


int get_most_bright_surface(MatrixXf mat, int row, int col)
{
    MatrixXf dst;

    int th = otsu_segmentation(mat);
    //cout << "seuil " << th << endl;
    binarize_image(mat,  dst , th , 0 , 255);



    std::vector<markers> list_markers;

    Vector3i color_value;

    size_t index = 0;

    markers mark;


    // take the point in format (y,x)
    mark.push_back(Vector2i(row, col));
    populate_marker_segmentation(mark, row, col, dst , index);
    list_markers.push_back(mark);

    return mark.size();
}



void dense_ground_labelling(string experience_name,string sub_experience_name, int scale_factor = 4, int wind_gravity_center = 6, bool sol = true)
{

     

    string inputFilename;

    std::vector<string> imageListCam1, imageListCam2;


    string path_to_ground_camera1 = string("..//").append(experience_name).append("//").append(experience_name).append(sub_experience_name).append(string("//Camera1"));
    string path_to_ground_camera2 = string("..//").append(experience_name).append("//").append(experience_name).append(sub_experience_name).append(string("//Camera2"));


    imageListCam1 = get_list_of_files(path_to_ground_camera1);
    imageListCam2 = get_list_of_files(path_to_ground_camera2);

    stereo_params_cv stereo_par;
    stereo_par.retreive_values();

    size_t nimages = imageListCam1.size();

    cout << "number of pairs for ground detection  " << nimages << endl ;
    cout << "-------------------------------------------------------------------------" << endl << endl << endl;

    std::list<stereo_match> matchings_stereo;
    phase_congruency_result<MatrixXf> pcr;


    std::vector<Vector3d> list_of_3d_points;


    string dir_name("..//Labelled//");



    for(int depth_dir = 0; depth_dir < 3 ; depth_dir++)
    {
        if(depth_dir == 0)
            create_directory(dir_name);

        if(depth_dir == 1)
        {
            string path_to_ = string(dir_name).append("//").append(experience_name) ;

            create_directory(path_to_);

            create_directory(path_to_+"//left");
            create_directory(path_to_+"//right");

            string path_to_test = string(dir_name).append("//").append(experience_name).append("//test");
            string path_to_train = string(dir_name).append("//").append(experience_name).append("//train");

            create_directory(path_to_test);
            create_directory(path_to_train);
        }


        if(depth_dir == 2)
        {
            //test
            {
                string path_to_left = string(dir_name).append("//").append(experience_name).append("//test").append("//left") ;
                string path_to_right = string(dir_name).append("//").append(experience_name).append("//test").append("//right");

                create_directory(path_to_left);
                create_directory(path_to_right);
            }

            //train
            {
                string path_to_left = string(dir_name).append("//").append(experience_name).append("//train").append("//left") ;
                string path_to_right = string(dir_name).append("//").append(experience_name).append("//train").append("//right");

                create_directory(path_to_left);
                create_directory(path_to_right);
            }

        }
    }


    string path_to_all = string(dir_name).append("//").append(experience_name);

    string path_to_test = string(dir_name).append("//").append(experience_name).append("//test");
    string path_to_train = string(dir_name).append("//").append(experience_name).append("//train");

    /********************************************************/


    std::string all_temp = path_to_all;
    all_temp = all_temp + "//all_log.txt";
    ofstream all_log (all_temp,  std::ios_base::app);

    std::string all_temp_gc = path_to_all;
    all_temp_gc = all_temp_gc + "//all_log_gc.txt";
    ofstream all_log_gc (all_temp_gc,  std::ios_base::app);

    /********************************************************/

    std::string test_temp = path_to_test;
    test_temp = test_temp + "//test_log.txt";
    ofstream test_log (test_temp,  std::ios_base::app);

    std::string train_temp = path_to_train;
    train_temp = train_temp + "//train_log.txt";
    ofstream train_log (train_temp, std::ios_base::app);


    /*********************************************************/

    std::string test_temp_svm = path_to_test;
    test_temp_svm = test_temp_svm + "//test_log_svm.txt";
    ofstream test_log_svm (test_temp_svm,  std::ios_base::app);

    std::string train_temp_svm = path_to_train;
    train_temp_svm = train_temp_svm + "//train_log_svm.txt";
    ofstream train_log_svm (train_temp_svm, std::ios_base::app);


    /**********************************************************/

    std::string test_temp_svm_gc = path_to_test;
    test_temp_svm_gc = test_temp_svm_gc + "//test_log_svm_gc.txt";
    ofstream test_log_svm_gc (test_temp_svm_gc,  std::ios_base::app);

    std::string train_temp_svm_gc = path_to_train;
    train_temp_svm_gc = train_temp_svm_gc + "//train_log_svm_gc.txt";
    ofstream train_log_svm_gc (train_temp_svm_gc, std::ios_base::app);

    /***********************************************************/

    if(test_log.is_open() && train_log.is_open())
    {

    }




    for(size_t index_img = 0 ;index_img < nimages ; index_img++)
    {

        //cout << "image pair " << index_img << endl;

        Mat left_view_16;
        Mat left_view;

        Mat right_view_16;
        Mat right_view;

        left_view_16 = imread(imageListCam1[index_img], CV_16U);

        double minVal, maxVal;
        Point minLoc, maxLoc;
        minMaxLoc(left_view_16, &minVal, &maxVal, &minLoc, &maxLoc);
        left_view_16.convertTo(left_view, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));

        if(scale_factor > 1)
            cv::resize(left_view, left_view, cv::Size(), scale_factor, scale_factor , INTER_CUBIC );

        minMaxLoc(left_view, &minVal, &maxVal, &minLoc, &maxLoc);
        Point2f coord_gc_left, coord_left;
        coord_left = maxLoc;
        get_local_maxima_gc(wind_gravity_center, left_view, maxLoc, coord_gc_left);

        right_view_16 = imread(imageListCam2[index_img], CV_16U);
        minMaxLoc(right_view_16, &minVal, &maxVal, &minLoc, &maxLoc);
        right_view_16.convertTo(right_view, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));

        if(scale_factor > 1)
            cv::resize(right_view, right_view, cv::Size(), scale_factor, scale_factor , INTER_CUBIC );

        int ncols = left_view.cols;
        int nrows = left_view.rows;


        minMaxLoc(right_view, &minVal, &maxVal, &minLoc, &maxLoc);
        Point2f coord_gc_right, coord_right;
        coord_right = maxLoc;
        get_local_maxima_gc(wind_gravity_center, right_view, maxLoc, coord_gc_right);

        //if( fabs(coord_right.y - coord_left.y) < 5 )
        {


            Vector2d left_pt;
            Vector2d right_pt;

            left_pt = Vector2d(coord_gc_left.y, coord_gc_left.x);
            right_pt = Vector2d(coord_gc_right.y, coord_gc_right.x);

            stereo_match st(left_pt, right_pt, 0);
            matchings_stereo.push_back(st);

            if(coord_gc_right.y - coord_gc_left.y > 10 || coord_gc_right.x - coord_gc_left.x > 10 )
            {
                continue;
            }

            MatrixXf left_eigen = mat_to_eigen(left_view);

            int left_surface = get_most_bright_surface(left_eigen, int(coord_gc_left.y), int(coord_gc_left.x));



            MatrixXf right_eigen = mat_to_eigen(right_view);

            int right_surface = get_most_bright_surface(right_eigen, int(coord_gc_right.y), int(coord_gc_right.x));

            cout << " left surface " << left_surface << "  right surface "  << right_surface << endl;

            if(left_surface > 20)
                continue;

            if(right_surface > 20)
                continue;

            left_view = eigen_to_mat(left_eigen);

            right_view = eigen_to_mat(right_eigen);




            Mat left_dot = Mat::zeros(nrows, ncols, left_view.type());

            for(int row = coord_left.y - wind_gravity_center ; row <= coord_left.y + wind_gravity_center; row++ )
            {
                if(row >= 0 && row < nrows)
                    for(int col = coord_left.x - wind_gravity_center; col <= coord_left.x + wind_gravity_center; col++ )
                    {
                        if(col >=0 && col < ncols)
                        {
                            left_dot.at<uchar>(row,col) = left_view.at<uchar>(row,col);
                        }
                    }
            }


            Mat right_dot = Mat::zeros(nrows, ncols, right_view.type());

            for(int row = coord_right.y - wind_gravity_center ; row <= coord_right.y + wind_gravity_center; row++ )
            {
                if(row >= 0 && row < nrows)
                    for(int col = coord_right.x - wind_gravity_center; col <= coord_right.x + wind_gravity_center; col++ )
                    {
                        if(col >=0 && col < ncols)
                        {
                            right_dot.at<uchar>(row,col) = right_view.at<uchar>(row,col);
                        }
                    }
            }


            cvtColor(left_dot, left_dot, COLOR_GRAY2RGB);
            cvtColor(right_dot, right_dot, COLOR_GRAY2RGB);

            //left_view.at<Vec3b>(Point(int(coord_left.x), int(coord_left.y) )) = Vec3b(125, 125,255);

            //right_view.at<Vec3b>(Point(int(coord_right.x), int(coord_right.y) )) = Vec3b(125, 125,255);

            string left_file_name;
            string right_file_name;

            string left_file_name_all;
            string right_file_name_all;

            if(sol)
            {
                left_file_name_all = string(path_to_all).append("//left//").append(to_string(index_img)).append("_sol_.png");
                right_file_name_all = string(path_to_all).append("//right//").append(to_string(index_img)).append("_sol_.png");
            }
            else
            {
                left_file_name_all = string(path_to_all).append("//left//").append(to_string(index_img)).append(".png");
                right_file_name_all = string(path_to_all).append("//right//").append(to_string(index_img)).append(".png");
            }


            srand(time(NULL));

            int num = rand() % 10 + 0;

            all_log_gc << std::fixed << std::setprecision(5) << index_img <<";" << coord_gc_left.x << ";" << coord_gc_left.y << ";" << coord_gc_right.x << ";" << coord_gc_right.y << ";" << sol  << endl;
            all_log << std::fixed << std::setprecision(5)  << index_img <<";" <<  coord_left.x << ";" << coord_left.y << ";" << coord_right.x << ";" << coord_right.y << ";" << sol  << endl;



            if(num < 2)
            {

                string p_left = string(path_to_test);
                string p_right = string(path_to_test);


                p_left.append(("//left"));
                p_right.append("//right");

                if(sol)
                {
                    left_file_name = string(p_left).append("//").append(to_string(index_img)).append("_sol_.png");
                    right_file_name = string(p_right).append("//").append(to_string(index_img)).append("_sol_.png");
                }
                else
                {
                    left_file_name = string(p_left).append("//").append(to_string(index_img)).append(".png");
                    right_file_name = string(p_right).append("//").append(to_string(index_img)).append(".png");
                }
                test_log_svm_gc << std::fixed << std::setprecision(5) << coord_gc_left.x << ";" << coord_gc_left.y << ";" << coord_gc_right.x << ";" << coord_gc_right.y << ";" << sol  << endl;
                test_log_svm << std::fixed << std::setprecision(5)  << coord_left.x << ";" << coord_left.y << ";" << coord_right.x << ";" << coord_right.y << ";" << sol  << endl;
                test_log << std::fixed << std::setprecision(5) << index_img << ";"
                                                                               "[" << coord_left.x << "|" << coord_left.y << "];"
                                                                                                                             "[" << coord_right.x << "|" << coord_right.y << "];"
                                                                                                                                                                             "[" << coord_gc_left.x << "|" << coord_gc_left.y << "];"
                                                                                                                                                                                                                                 "[" << coord_gc_right.x << "|" << coord_gc_right.y << "];"
                                                                                                                                                                                                                                                                                       "" << sol  << endl;
            }
            else
            {

                string p_left = string(path_to_train);
                string p_right = string(path_to_train);


                p_left.append(("//left"));
                p_right.append("//right");

                if(sol)
                {
                    left_file_name = string(p_left).append("//").append(to_string(index_img)).append("_sol_.png");
                    right_file_name = string(p_right).append("//").append(to_string(index_img)).append("_sol_.png");
                }
                else
                {
                    left_file_name = string(p_left).append("//").append(to_string(index_img)).append(".png");
                    right_file_name = string(p_right).append("//").append(to_string(index_img)).append(".png");
                }
                //train_log << std::fixed << std::setprecision(5) << index_img << ";" << coord_gc_left.x << ";" << coord_gc_left.y << ";" << coord_gc_right.x << ";" << coord_gc_right.y  << ";" << sol << endl;
                train_log_svm_gc << std::fixed << std::setprecision(5) << coord_gc_left.x << ";" << coord_gc_left.y << ";" << coord_gc_right.x << ";" << coord_gc_right.y << ";" << sol  << endl;
                train_log_svm << std::fixed << std::setprecision(5)  << coord_left.x << ";" << coord_left.y << ";" << coord_right.x << ";" << coord_right.y << ";" << sol  << endl;
                train_log << std::fixed << std::setprecision(5) << index_img << ";"
                                                                                "[" << coord_left.x << "|" << coord_left.y << "];"
                                                                                                                              "[" << coord_right.x << "|" << coord_right.y << "];"
                                                                                                                                                                              "[" << coord_gc_left.x << "|" << coord_gc_left.y << "];"
                                                                                                                                                                                                                                  "[" << coord_gc_right.x << "|" << coord_gc_right.y << "];"
                                                                                                                                                                                                                                                                                        "" << sol  << endl;
            }

            //cv::resize(right_view, right_view, cv::Size(), 8, 8 , INTER_CUBIC );
            //cv::resize(left_view, left_view, cv::Size(), 8, 8 , INTER_CUBIC );

            imwrite(left_file_name, left_dot);
            imwrite(right_file_name, right_dot);

            imwrite(left_file_name_all, left_dot);
            imwrite(right_file_name_all, right_dot);

            //cout << "Points matching before : " << coord_left  << "  " << coord_right <<
            //        "      Points matching after : " << coord_gc_left  << "  " << coord_gc_right << "side " <<  num << endl;

        }

    }

}






}
