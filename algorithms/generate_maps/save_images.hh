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
#include "global_var.hh"
#include <experimental/filesystem>
#include "core.hpp"
#include "algorithms/miscellaneous.hh"
#include "global_utils.hh"


using namespace std;
using namespace Eigen;
 

namespace tpp
{


string get_number_of_zero(int value, int number_of_zeros = 6)
{
    string nb_zeros = "";
    if(value > 0)
    {
        number_of_zeros = number_of_zeros - int(floor(log10(value))) ;
    }
    else
    {
        number_of_zeros = number_of_zeros - int(floor(log10(value+1))) ;
    }
    for(int i = 0; i < number_of_zeros; i++)
        nb_zeros = nb_zeros + "0";

    //cout << number_of_zeros << "  " << value  << "  "  << nb_zeros << endl;
    return nb_zeros;
}


void save_image_from_z(projected_objects_image list_objects, MatrixXf left_eigen,
                       int nrows, int ncols , size_t index_img, string tof,
                       float alpha = 0.1, string place_to_save = "..//Results//",
                       string experience_name = "_", string sub_experience_name ="_", float threshold = 0.1,
                       ColorMapType cm = ColorMapType::COLOR_MAP_TYPE_VIRIDIS)
{

    //cout << "save_image_from_z " << list_objects.size() << endl;
    string where_to_save = place_to_save;
    create_directory(where_to_save);
    where_to_save = where_to_save + "//" + tof;
    create_directory(where_to_save);
    where_to_save = where_to_save + "//" + experience_name;
    create_directory(where_to_save);
    where_to_save = where_to_save + "//" + sub_experience_name;
    create_directory(where_to_save);
    where_to_save = where_to_save + "//" + std::to_string(threshold);
    create_directory(where_to_save);

    //std::string tm =  where_to_save + "//" + std::to_string(10000+ index_img) + "_" + tof + "_list_matches.txt";
    //cout << tm << endl;
    //ofstream log_matches (tm,  std::ios_base::app);


    alpha = 0;

    MatrixXf results_3d = MatrixXf::Zero(nrows, ncols);
    Matrix<Vector3d, Dynamic, Dynamic> colored_img(nrows, ncols);

    for(int row = 0 ; row < nrows; row++)
    {
        for(int col = 0; col < ncols ; col ++)
        {
            colored_img(row, col) = Vector3d(0, 0, 0);
        }
    }


    for(size_t idx_object = 0 ; idx_object < list_objects.size() ; idx_object++ )
    {
        for(size_t idx_projection = 0; idx_projection < list_objects[idx_object].size(); idx_projection++)
        {

            //cout << "mlfkfjklfnjkfgjfjkl " << list_objects.size() << endl;

            //if (list_objects[idx_object][idx_projection].point_projected(0) > 100)
            {
                /*log_matches << std::fixed << std::setprecision(5)  << list_objects[idx_object][idx_projection].first_point(0) << ";"
                            << list_objects[idx_object][idx_projection].first_point(1) << ";" << list_objects[idx_object][idx_projection].second_point(0) << ";" <<list_objects[idx_object][idx_projection].second_point(1) << ";"
                            << list_objects[idx_object][idx_projection].point_projected(0) << ";" << list_objects[idx_object][idx_projection].point_projected(1) << ";" << list_objects[idx_object][idx_projection].point_projected(2)  << endl;
                    */

                /*cout << list_objects[idx_object][idx_projection].first_point.y() << ";"
                     << list_objects[idx_object][idx_projection].first_point.x() << ";" << list_objects[idx_object][idx_projection].second_point.y() << ";" <<list_objects[idx_object][idx_projection].second_point.x() << ";"
                     << list_objects[idx_object][idx_projection].point_projected.x() << ";" << list_objects[idx_object][idx_projection].point_projected.y() << ";" << list_objects[idx_object][idx_projection].point_projected.z()  << endl;
                */
                results_3d(int(list_objects[idx_object][idx_projection].first_point.y()), int(list_objects[idx_object][idx_projection].first_point.x())) = fabs(list_objects[idx_object][idx_projection].point_projected.z());

            }
        }
    }





    /*results_3d(0,0) = -10000;
    results_3d(nrows-1,ncols-1) = 10000;*/

    normalize_matrix(results_3d, 1);

    /*results_3d(0,0) = 0;
    results_3d(nrows-1,ncols-1) = 0;*/



    Matrix<Vector3d, Dynamic, Dynamic> blent_img(nrows, ncols);

    float beta;

    beta = ( 1.0 - alpha );

    bool blending = false;

    for(int row = 0 ; row < nrows; row++)
    {
        for(int col = 0; col < ncols ; col ++)
        {
            if (blending)
            {
                float r, g , b;
                colormap<float>( ColorMapType::COLOR_MAP_TYPE_VIRIDIS, results_3d(row , col), r, g, b  );

                colored_img(row,col) = 255.0 * Vector3d(r, g, b);
                //Vector3i color = generate_color(500);
                //colored_img(row,col) = Vector3d(255, 255, 255);
                float new_red = alpha * left_eigen(row, col) + beta * 255 * r;
                float new_green = alpha * left_eigen(row, col) + beta * 255 * g;
                float new_blue = alpha * left_eigen(row, col) + beta * 255 * b;

                blent_img(row, col) = Vector3d(new_red, new_green, new_blue);
            }
            else
            {
                if(results_3d(row , col) > 0)
                    colored_img(row,col) = Vector3d(255, 255, 255);
                else
                    colored_img(row,col) = Vector3d(0, 0, 0);
            }
        }
    }

    Mat results_3d_open_cv;

    if(blending)
    {
        results_3d_open_cv =  eigen_to_mat_template<Vector3d>(blent_img);
    }
    else
    {
        results_3d_open_cv =  eigen_to_mat_template<Vector3d>(colored_img);
    }



    string image_name = string(where_to_save + "//");
    image_name = image_name + tof + "_" + get_number_of_zero(index_img) + to_string(index_img) + ".png";

    //cv::resize(results_3d_open_cv, results_3d_open_cv, cv::Size(), 10, 10 , INTER_CUBIC );

    imwrite(image_name, results_3d_open_cv);
}


void save_pc_images(phase_congruency_result<MatrixXf> pcr, int cp, string place_to_save = "..//ResultsPC//",
                    string experience_name = "_", string sub_experience_name = "_")
{
    MatrixXf PC_left = pcr.pc_first;
    //cout << PC_left << endl;
    MatrixXf PC_right = pcr.pc_second;

    string where_to_save = place_to_save;
    create_directory(where_to_save);
    where_to_save = where_to_save + experience_name + "//" ;
    create_directory(where_to_save);
    where_to_save = where_to_save + sub_experience_name + "//" ;
    create_directory(where_to_save);

    string image_name = where_to_save;
    image_name.append(string("left_")).append(get_number_of_zero(cp) + to_string(cp)).append(string(".png"));


    normalize_matrix(PC_left,  255);
    Mat res_l = eigen_to_mat(PC_left);
    imwrite(image_name,res_l);

    image_name = where_to_save;
    image_name.append(string("right_")).append(get_number_of_zero(cp) + to_string(cp)).append(string(".png"));

    normalize_matrix(PC_right,  255);
    Mat res_r = eigen_to_mat(PC_right);
    imwrite(image_name,res_r);
}




void save_ori_images(cv::Mat left_img, cv::Mat right_img, int cp, string place_to_save = "..//Oricy//",
                     string experience_name = "_", string sub_experience_name = "_", bool label_image = true, bool label_name = true)
{

    // 	 x = (condition) ? (value_if_true) : (value_if_false);

    //string complet_name = (label_image) ? ("_sol_") : ("_haut_");

    //cv::resize(left_img, left_img, cv::Size(), 8, 8 , INTER_CUBIC );
    //cv::resize(left_img, left_img, cv::Size(), 8, 8 , INTER_CUBIC );

    string where_to_save = place_to_save;
    create_directory(where_to_save);
    where_to_save = where_to_save + experience_name + "//" ;
    create_directory(where_to_save);
    where_to_save = where_to_save + sub_experience_name + "//" ;
    create_directory(where_to_save);

    srand(time(NULL));
    int num = rand() % 10 + 0;

    /*if (num > 2)
    {
        where_to_save = where_to_save +  "train//" ;
        create_directory(where_to_save);
    }
    else
    {
        where_to_save = where_to_save +  "test//" ;
        create_directory(where_to_save);
    }*/

    create_directory(where_to_save + "/left");
    create_directory(where_to_save + "/right");

    string image_name = where_to_save + "/left/";

    image_name.append(string("left_")).append(get_number_of_zero(cp) + to_string(cp)).append( string(".png"));
    imwrite(image_name,left_img);


    image_name = where_to_save + "/right/";
    image_name.append(string("right_")).append(get_number_of_zero(cp) + to_string(cp)).append( string(".png"));
    imwrite(image_name,right_img);
}

void save_stereo_segmentation_results(MatrixXf left_image_seg, MatrixXf right_image_seg, float eps, std::vector<stereo_match> epipolar_order_constraints,int cp=0,
                                      string place_to_save = "..//Segmentation_results//", string experience_name = "_", string sub_experience_name="_")
{

    cout << "save_stereo_images_matching_segmented"  << endl;

    int ncols = left_image_seg.cols();
    int nrows = left_image_seg.rows();


    Mat left_mat = eigen_to_mat(left_image_seg,1);
    Mat right_mat = eigen_to_mat(right_image_seg,1);

    //left_mat = cv::Mat::zeros(nrows, ncols, CV_8U );
    //right_mat = cv::Mat::zeros(nrows, ncols, CV_8U );

    //cout << "ggr " << endl;
    cvtColor(left_mat, left_mat, cv::COLOR_GRAY2BGR);
    cvtColor(right_mat, right_mat, cv::COLOR_GRAY2BGR);

    //cout << "taille " << nrows << "  " << ncols << endl;

    //cout << "taille " << left_mat.rows << "  " << left_mat.cols << endl;

    int nb = 0;

    if(epipolar_order_constraints.size() >= 1)
    {
        for (auto &l : epipolar_order_constraints)
        {
            stereo_match smt = l;
            //if( fabs(smt.left_point.y() - smt.right_point.y()) < 10 && nb )
            {
                //break;

                //if (smt.similarity <= 0.1)
                //    break;

                Point pt1;
                pt1.y = int(smt.first_point.y());
                pt1.x = int(smt.first_point.x());
                Point pt2;
                pt2.y = int(smt.second_point.y());
                pt2.x = int(smt.second_point.x());

                Point pt_;
                pt_.y = int(smt.second_point.y());
                pt_.x = int(smt.second_point.x());
                Vector3i color = generate_color(500);

                //if(nb==10)
                {
                    left_mat.at<Vec3b>(pt1) = Vec3b(uchar(color[0]), uchar(color[1]), uchar(color[2]));
                    right_mat.at<Vec3b>(pt2) = Vec3b(uchar(color[0]), uchar(color[1]), uchar(color[2]));
                }




                nb++;
                if (nb > 100000)
                    break;
            }
        }
    }


    //cv::resize(left_mat, left_mat, cv::Size(), 10, 10 , INTER_CUBIC );
    //cv::resize(right_mat, right_mat, cv::Size(), 10, 10 , INTER_CUBIC );

    string where_to_save = place_to_save;
    create_directory(where_to_save);
    where_to_save = where_to_save + experience_name + "//" ;
    create_directory(where_to_save);
    where_to_save = where_to_save + sub_experience_name + "//" ;
    create_directory(where_to_save);

    std::string file_name_left = where_to_save;
    file_name_left = file_name_left.append(get_number_of_zero(cp) + to_string(cp)).append(std::string("_")).append(std::string("left")).append(std::string("_sim.png"));
    imwrite(file_name_left, left_mat);

    std::string file_name_right = where_to_save;
    file_name_right = file_name_right.append(get_number_of_zero(cp) + to_string(cp)).append(std::string("_")).append(std::string("right")).append(std::string("_sim.png"));
    imwrite(file_name_right, right_mat);


}



void save_stereo_draw_matches(MatrixXf left_image, MatrixXf right_image, float eps, std::vector<stereo_match> epipolar_order_constraints, string dir_m ,int cp=0)
{


    int ncols = left_image.cols();
    int nrows = left_image.rows();

    MatrixXf img_draw = MatrixXf::Zero(2*nrows, 2*ncols);

    img_draw.block(0, 0, nrows, ncols) = left_image;
    img_draw.block(nrows, ncols, nrows, ncols) = right_image;


    Mat left_mat = eigen_to_mat(left_image);
    Mat right_mat = eigen_to_mat(right_image);

    Mat img_draw_combined = Mat::zeros(img_draw.rows(), img_draw.cols(), CV_8U);

    for (int row = 0; row < img_draw.rows(); row++)
    {
        for (int col = 0; col < img_draw.cols(); col++)
        {
            img_draw_combined.at<uchar>(row, col) = uchar(int(img_draw(row, col)));

        }
    }
    cvtColor(img_draw_combined, img_draw_combined, cv::COLOR_GRAY2BGR);
    cvtColor(left_mat, left_mat, cv::COLOR_GRAY2BGR);
    cvtColor(right_mat, right_mat, cv::COLOR_GRAY2BGR);

    //cout << "pomme " << epipolar_order_constraints.size() << endl;

    int nb = 0;
    if(epipolar_order_constraints.size() >= 1)
    {
        for (auto &l : epipolar_order_constraints)
        {
            stereo_match smt = l;
            //if( fabs(smt.left_point.y() - smt.right_point.y()) < 10 && nb )
            {
                //break;

                //cout << "save_stereo_draw_matches  " << nb << endl;

                if (smt.similarity <= 0.1)
                    break;

                Point pt_left;
                pt_left.y = int(smt.first_point.y());
                pt_left.x = int(smt.first_point.x());
                Point pt_right;
                pt_right.y = int(smt.second_point.y());
                pt_right.x = int(smt.second_point.x());

                Point pt_combined;
                pt_combined.y = nrows + int(smt.second_point.y());
                pt_combined.x = ncols + int(smt.second_point.x());

                Vector3i color = generate_color(500);
                line(img_draw_combined, pt_left, pt_combined, cv::Scalar(color[0], color[1], color[2]), 1);

                left_mat.at<Vec3b>(pt_left) = Vec3b(color[0], color[1], color[2]);
                right_mat.at<Vec3b>(pt_right) = Vec3b(color[0], color[1], color[2]);

                nb++;
                if (nb > 100000)
                    break;
            }
        }
    }

    std::string file_left = dir_m + "//";
    file_left = file_left.append(get_number_of_zero(cp) + to_string(cp)).append(std::string("_")).append(std::string("_left_matches.png"));
    imwrite(file_left, left_mat);

    std::string file_name_right = dir_m + "//";
    file_name_right = file_name_right.append(get_number_of_zero(cp) + to_string(cp)).append(std::string("_")).append(std::string("_right_matches.png"));
    imwrite(file_name_right, right_mat);



    std::string file_name_combined = dir_m + "//";
    file_name_combined = file_name_combined.append(get_number_of_zero(cp) + to_string(cp)).append(std::string("_")).append(std::string("_combined_matches.png"));
    imwrite(file_name_combined, img_draw_combined);



}

void save_stereo_images_markers(std::vector<markers> list_markers, std::vector<stereo_match> list_projection, int nrows, int ncols, int cp=0, bool verbose = false,
                                string place_to_save =  "..//Matching_on//", string experience_name = "_", string sub_experience_name = "_")
{

    /*if(verbose)
        cout << "save_stereo_images_markers start"  << endl;*/

    if(list_projection.size() < 1)
        return;

    Mat left_mat = Mat::zeros(nrows, ncols , CV_8U);

    cvtColor(left_mat, left_mat, cv::COLOR_GRAY2BGR);

    int nb = 0;
    for(int i = 0 ; i < list_markers.size() ; i ++)
    {
        for(int j = 0 ; j < list_markers[i].size(); j++)
        {
            left_mat.at<Vec3b>(list_markers[i][j].y(), list_markers[i][j].x()) = Vec3b(255, 255, 255);
        }
    }

    for(int i = 0 ; i < list_projection.size() ; i ++)
    {
        Vector3i color = generate_color(500);
        left_mat.at<Vec3b>(list_projection[i].first_point.y(), list_projection[i].first_point.x()) = Vec3b(color[0], color[1], color[2]);

    }
    string where_to_save = place_to_save;
    create_directory(where_to_save);
    where_to_save = where_to_save + experience_name + "//" ;
    create_directory(where_to_save);
    where_to_save = where_to_save + sub_experience_name + "//" ;
    create_directory(where_to_save);
    std::string file_name_left = where_to_save;
    file_name_left = file_name_left.append(get_number_of_zero(cp) + to_string(cp)).append(std::string("_")).append(std::string("left")).append(std::string("_marker.png"));

    //cv::resize(left_mat, left_mat, cv::Size(), 8, 8 , INTER_CUBIC );
    imwrite(file_name_left, left_mat);

    /*if(verbose)
        cout << "save_stereo_images_markers end"  << endl;*/


}


}
