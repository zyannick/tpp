#pragma once

#include "algorithms/generate_maps/sparse_disparity_maps.hh"
#include "algorithms/segmentation.hh"
#include "algorithms/fitting.hh"
#include "global_utils.hh"
#include "algorithms/kernels.hh"
#include "core.hpp"
#include "save_images.hh"


namespace tpp {


void correct_coords(int gaussian_x, int gaussian_y, Vector2i point_ , int mid_x, int mid_y ,int nrows, int ncols ,
                    int &coord_x, int &coord_y, int &decal_x, int &decal_y, int &new_gaussian_x, int &new_gaussian_y)
{
    coord_x = 0;
    coord_y = 0;
    decal_x = 0;
    decal_y = 0;
    new_gaussian_x = gaussian_x;
    new_gaussian_y = gaussian_y;

    coord_x = point_.x() - mid_x;
    if(coord_x < 0)
    {
        decal_x = coord_x;
        new_gaussian_x = gaussian_x + decal_x;
        coord_x = 0;
        decal_x = - decal_x;
    }
    coord_y = point_.y() - mid_y;
    if(coord_y < 0)
    {
        decal_y = coord_y;
        new_gaussian_y = gaussian_y + decal_y;
        coord_y = 0;
        decal_y = -decal_y;
    }

    if( gaussian_x + coord_x >= ncols)
    {
        new_gaussian_x = gaussian_x - ( gaussian_x + coord_x - ncols);
    }

    if(gaussian_y + coord_y >= nrows)
    {
        new_gaussian_y = gaussian_y - ( gaussian_y + coord_y - nrows);
    }
}

template< typename T>
std::pair< Matrix<T, Dynamic, Dynamic> , Matrix<T, Dynamic, Dynamic>  >
generate_view_from_two_points(Vector2i first_point, Vector2i second_point, int gaussian_x, int gaussian_y, int nrows,int ncols)
{
    //cout << "generate_view_from_two_points" << endl;

    Matrix<T, Dynamic, Dynamic> kernel_ = gaussian_kernel_iso<T>(gaussian_x, gaussian_y, 3);
    normalize_matrix_template<T>(kernel_);
    //cout << "kernel okay"  << endl;
    Matrix<T, Dynamic , Dynamic> left_img = Matrix<T, Dynamic , Dynamic>::Zero(nrows, ncols);
    Matrix<T, Dynamic , Dynamic> right_img = Matrix<T, Dynamic , Dynamic>::Zero(nrows, ncols);
    Matrix<T, Dynamic, Dynamic> gaussian_multiply = kernel_ * T(255);

    //cout << "multiply kernel okay " << first_point.transpose() << "    " << second_point.transpose()  <<endl;

    int mid_x = gaussian_x / 2;
    int mid_y = gaussian_y / 2;

    int coord_x;
    int coord_y;
    int decal_x;
    int decal_y;
    int new_gaussian_x;
    int new_gaussian_y;

    correct_coords(gaussian_x, gaussian_y, first_point , mid_x, mid_y ,nrows, ncols ,
                   coord_x, coord_y, decal_x, decal_y, new_gaussian_x, new_gaussian_y);

    //cout << "nouvelles valeurs left  " << decal_y  << "   "  << decal_x  << "   " << new_gaussian_x << "   "  << new_gaussian_y << "   " << gaussian_x << "   "  << gaussian_y << "   " << coord_y   << "   " << coord_x  << endl;
    left_img.block(coord_y, coord_x, new_gaussian_y, new_gaussian_x ) = gaussian_multiply.block(decal_y,  decal_x, new_gaussian_y,  new_gaussian_x );

    correct_coords(gaussian_x, gaussian_y, second_point , mid_x, mid_y ,nrows, ncols ,
                   coord_x, coord_y, decal_x, decal_y, new_gaussian_x, new_gaussian_y);

    //cout << "nouvelles valeurs right  " << decal_y  << "   "  << decal_x  << "   " << new_gaussian_x << "   "  << new_gaussian_y << "   " << coord_y   << "   " << coord_x  << endl;

    right_img.block(coord_y, coord_x, new_gaussian_y, new_gaussian_x ) = gaussian_multiply.block(decal_y,  decal_x, new_gaussian_y,  new_gaussian_x );

    auto pom = std::pair(left_img, right_img);

    //cout << "generate_view_from_two_points end" << endl;

    return pom;

}



template<typename T>
int generate_view_from_list_of_two_points(projected_objects_image list_objects, int nrows, int ncols, string path_to,
                                           string path_to_left, string path_to_right, int index_img,bool verbose)
{
    if(verbose)
    {
        cout << "generate_view_from_list_of_two_points " << list_objects.size()  << endl;
    }

    std::string path_to_detect = path_to;
    path_to_detect = path_to_detect + "//detected.csv";
    ofstream log_detect (path_to_detect,  std::ios_base::app);

    if(list_objects.size() < 1)
    {
        log_detect << std::fixed << std::setprecision(5) << 0 << ";" << -1
                   << ";" << -1
                   << ";" << -1
                   << ";" << -1
                   << endl ;
        return 0;
    }

    int nb_matches = 0;

    int id_to_retrieve = 0;



    for(int idx_object = 0 ; idx_object < list_objects.size() ; idx_object++ )
    {
        if(verbose)
        {
            cout << "object number " <<  idx_object << "   " << list_objects[idx_object].size() << endl;
        }

        for(int idx_projection = 0; idx_projection < list_objects[idx_object].size(); idx_projection++)
        {
            //cout << "here 1" << endl;
            Vector2i first_point = list_objects[idx_object][idx_projection].first_point.cast<int>();
            Vector2i second_point = list_objects[idx_object][idx_projection].second_point.cast<int>();


            log_detect << std::fixed << std::setprecision(5) << id_to_retrieve
                       << ";" << idx_projection
                       << ";" << list_objects[idx_object][idx_projection].first_point.x()
                       << ";" << list_objects[idx_object][idx_projection].first_point.y()
                       << ";" << list_objects[idx_object][idx_projection].second_point.x()
                       << ";" << list_objects[idx_object][idx_projection].second_point.y()
                       << endl ;

            nb_matches++;

            std::pair< Matrix<T, Dynamic, Dynamic> , Matrix<T, Dynamic, Dynamic>  > view_pairs = generate_view_from_two_points<T>(first_point, second_point, 3, 3, nrows, ncols);

            //cout << "here 3" << endl;
            cv::Mat left_view = eigen_to_mat_template_1d<T>(view_pairs.first);
            cv::Mat right_view = eigen_to_mat_template_1d<T>(view_pairs.second);
            //cout << "here 4" << endl;

            cvtColor(left_view, left_view, COLOR_GRAY2RGB);
            cvtColor(right_view, right_view, COLOR_GRAY2RGB);

            //cout << "here 5" << endl;

            string p_l =  string(path_to_left) + "//" + get_number_of_zero(id_to_retrieve) + to_string(id_to_retrieve) + "_" + get_number_of_zero(index_img) + to_string(index_img) + "_" + to_string(idx_object+1) + "_" + to_string(idx_projection+1) + "_.png";

            string p_r =  string(path_to_right) + "//" + get_number_of_zero(id_to_retrieve) + to_string(id_to_retrieve) + "_" + get_number_of_zero(index_img) + to_string(index_img) + "_" + to_string(idx_object+1) + "_" + to_string(idx_projection+1) + "_.png";

            //cout << "here 6" << endl;

            imwrite(p_l, left_view);
            imwrite(p_r, right_view);
            id_to_retrieve++;
        }
    }

    if(verbose)
    {
        //cout << "generate_view_from_list_of_two_points end" << list_objects.size()  << endl;
    }

    return nb_matches;

}



template<typename T>
void generate_view_from_list_of_two_points_motion(projected_objects_image list_objects, int nrows, int ncols, string path_to, string path_to_left, string path_to_right, int index_img, bool verbose)
{
    if(verbose)
    {
        cout << "generate_view_from_list_of_two_points_motion " << list_objects.size()  << endl;
    }

    if(list_objects.size() < 1)
    {
        return;
    }
    std::string path_to_detect = path_to;
    path_to_detect = path_to_detect + "//detected.csv";
    ofstream log_detect (path_to_detect,  std::ios_base::app);


    for(int i = 0 ; i < list_objects.size() ; i++ )
    {
        if(verbose)
        {
            cout << "object number " <<  i << "   " << list_objects[i].size() << endl;
        }

        for(int j = 0; j < list_objects[i].size(); j++)
        {
            Vector2i first_point = list_objects[i][j].first_point.cast<int>();
            Vector2i second_point = list_objects[i][j].second_point.cast<int>();


            log_detect << std::fixed << std::setprecision(5) << j << ";" << list_objects[i][j].first_point(1) << ";" << list_objects[i][j].first_point(0)
                       << ";" << list_objects[i][j].second_point(1) << ";" << list_objects[i][j].second_point(0) << endl ;

            std::pair< Matrix<T, Dynamic, Dynamic> , Matrix<T, Dynamic, Dynamic>  > view_pairs = generate_view_from_two_points<T>(first_point, second_point, 3, 3, nrows, ncols);


            cv::Mat left_view = eigen_to_mat_template_1d<T>(view_pairs.first);
            cv::Mat right_view = eigen_to_mat_template_1d<T>(view_pairs.second);

            cvtColor(left_view, left_view, COLOR_GRAY2RGB);
            cvtColor(right_view, right_view, COLOR_GRAY2RGB);

            string p_l =  string(path_to_left) + "//_" + to_string(index_img) + "_" + to_string(i+1)
                    + "_" + to_string(j+1) + "_id" + to_string(list_objects[i][j].id) + "_depth" + to_string(list_objects[i][j].depth) + "_.png";

            string p_r =  string(path_to_right) + "//_" + to_string(index_img) + "_" + to_string(i+1)
                    + "_" + to_string(j+1) + "_id" + to_string(list_objects[i][j].id) + "_depth" + to_string(list_objects[i][j].depth) + "_.png";

            imwrite(p_l, left_view);
            imwrite(p_r, right_view);
        }
    }
}


}
