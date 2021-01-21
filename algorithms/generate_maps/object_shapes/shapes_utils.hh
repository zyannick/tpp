#pragma once

#include "algorithms/generate_maps/sparse_disparity_maps.hh"
#include "algorithms/segmentation.hh"
#include "algorithms/generate_maps.hh"
#include "algorithms/fitting.hh"
#include "global_utils.hh"
#include "algorithms/kernels.hh"
#include "core.hpp"


namespace tpp {


float distance_between_two_objects(projected_object object1, projected_object object2)
{
    float min_dist = 100000000.0;
    for(int i = 0; i < object1.size(); i ++ )
    {
        for(int j = 0; j < object2.size(); j++)
        {
            if ( (object1[i].first_point - object2[i].first_point).norm() < min_dist )
            {
                min_dist = (object1[i].first_point - object2[i].first_point).norm();
            }
        }
    }
    return min_dist;
}


/**
     * @brief merge_two_objects
     * @param object1
     * @param object2
     * @return
     */
projected_object merge_two_objects(projected_object object1, projected_object object2)
{
    for(int i = 0 ; i < object2.size(); i++)
    {
        object1.push_back(object2[i]);
    }
    return object1;
}


/**
     * @brief get_closest_object
     * @param list_objects
     * @param previous_mean_disp
     * @param threshold_disp
     * @param threshold_dist
     * @return
     */
Vector2d get_closest_object(projected_objects_image &list_objects, Vector2d previous_mean_disp = Vector2d(0,0), float threshold_disp = 0.5, float threshold_dist = 10, bool verbose = false)
{


    std::vector<Vector4d> list_of_disparity;
    for(int idx = 0 ; idx < list_objects.size(); idx++)
    {
        std::pair<float, float> pair_means =  get_mean_disparity( list_objects[idx]);
        Vector4d temp = Vector4d( idx, pair_means.first , pair_means.second, -1 );
        list_of_disparity.push_back(temp);
    }

    if(list_objects.size() < 2)
    {
        if(list_objects.size() < 1)
        {
            return Vector2d(0,0);
        }
        else
        {
            return list_of_disparity[0].segment(1,2);
        }
    }


    std::sort(begin(list_of_disparity), end(list_of_disparity), sort_mean_descending());

    projected_objects_image clustered_objects;

    bool to_merge = false;


    for(int i = 0; i < list_of_disparity.size() ; i++)
    {
        for(int j = 0 ; j < list_of_disparity.size() ; j++)
        {
            if( i!=j && list_of_disparity[j](3) == -1 &&
                    (list_of_disparity[i].segment(1,2) - list_of_disparity[j].segment(1,2)).norm() < threshold_disp )
            {
                float min_dist = distance_between_two_objects(list_objects[i], list_objects[j]);

                if(min_dist < threshold_dist)
                {
                    if(verbose)
                    cout << "[" << list_of_disparity[i](1)  << "," << list_of_disparity[i](2)
                                                               << "]      ["  << list_of_disparity[j](1)  << "," << list_of_disparity[j](2) << "]" << endl;
                    if(verbose)
                    cout << list_of_disparity[j](3)<< "  merge  " << i << " and " << j  << endl;
                    list_of_disparity[j](3) = i;
                    list_of_disparity[i](3) = j;


                    to_merge = true;
                    if(verbose)
                    cout << endl << endl;
                }
            }
        }
    }


    projected_object main_object;

    if(to_merge)
    {
        for(int i = 0; i < list_of_disparity.size() ; i++)
        {
            if(list_of_disparity[i](3) != -1)
            {
                int s = list_of_disparity[i](3);
                int f = i;

                main_object = merge_two_objects(list_objects[f], list_objects[s]);

            }
        }
    }

    if(main_object.size() == 0)
    {
        //unable to merge
        int si = -1;
        float dist_now = 100000;
        for(int i = 0; i < list_of_disparity.size() ; i++)
        {
            Vector2d val = list_of_disparity[i].segment(1,2);
            if( dist_now > (previous_mean_disp - val).norm())
            {
                if(si == -1)
                {
                    si = list_of_disparity[i](0);
                    Vector2d val = list_of_disparity[i].segment(1,2);
                    dist_now = (previous_mean_disp - val).norm();
                }
                else
                {
                    int ni = list_of_disparity[i](0);
                    if(list_objects[si].size()  < list_objects[ni].size())
                    {
                        si = ni;
                    }
                }
            }
        }
        if(si != -1)
        {
            main_object = list_objects[si];
        }
    }

    if(main_object.size() > 0)
    {
        clustered_objects.push_back(main_object);
    }

    list_objects = clustered_objects;

    std::pair<float, float> final_pair_means =  get_mean_disparity(main_object);

    return Vector2d(final_pair_means.first, final_pair_means.second);/**/

}


/**
     * @brief get_objects
     * @param list_markers
     * @param list_projection
     * @param surface_min_person
     * @param list_objects
     * @param list_correct_markers
     * @param min_l2_dist
     * @param merge_object
     * @return
     */
Vector2d get_objects(std::vector<markers> list_markers, std::vector<tpp::stereo_match> list_projection,
                     int surface_min_person, projected_objects_image &list_objects, std::vector<markers> &list_correct_markers,
                     int &nb_pixels, Vector2d mean_disp = Vector2d(0,0), bool verbose = false, int min_l2_dist = 2, bool merge_object = true)
{
    int min_distance = 10000;

    Vector2d new_mean_disp;

    for(int index_marker = 0 ; index_marker < list_markers.size() ; index_marker++)
    {
        markers mark = list_markers[index_marker];
        std::vector<stereo_match> temp_3d_points;

        for(int k = 0 ; k < mark.size() ; k++)
        {
            Vector2i val_comp = mark[k];
            for(int i = 0; i < list_projection.size(); i ++)
            {
                Vector2i val = (list_projection[i].first_point).cast<int>();
                float norm = (val_comp - val).cast<float>().norm();

                if(norm < min_l2_dist)
                {
                    if(list_projection[i].taken!=1)
                    {
                        temp_3d_points.push_back(list_projection[i]);
                        list_projection[i].taken = 1;
                    }
                }
            }
        }
        nb_pixels += mark.size();
        //cout << "temp_3d_points  " << temp_3d_points.size() << endl;
        if(temp_3d_points.size() > 4)
        {
            list_objects.push_back(temp_3d_points);
            list_correct_markers.push_back(list_markers[index_marker]);
        }
    }


    for(int idx_object = 0 ; idx_object < list_objects.size() ; idx_object++ )
    {
        if(verbose)
        {
            cout << "object number " <<  idx_object << "   " << list_objects[idx_object].size() << endl;
        }

        /*for(int idx_projection = 0; idx_projection < list_objects[idx_object].size(); idx_projection++)
        {
            Vector2i first_point = list_objects[idx_object][idx_projection].first_point.cast<int>();
            Vector2i second_point = list_objects[idx_object][idx_projection].second_point.cast<int>();

            cout << "list points " << list_objects[idx_object][idx_projection].first_point.x()
                 << ";" << list_objects[idx_object][idx_projection].first_point.y()
                 << ";" << list_objects[idx_object][idx_projection].second_point.x()
                 << ";" << list_objects[idx_object][idx_projection].second_point.y()
                 << ";" << list_objects[idx_object][idx_projection].point_projected.z() << endl ;

        }*/
    }

    if(merge_object)
    {
        new_mean_disp = get_closest_object(list_objects, mean_disp);
    }

    return new_mean_disp;
}

void get_objects_stereo_match(std::vector<markers> list_markers, std::vector<stereo_match> list_projection,
                              int surface_min_person,projected_objects_image &list_objects, std::vector<markers> &list_correct_markers , int min_l2_dist = 2, bool merge_object = true)
{
    int min_distance = 10000;

    for(int index_marker = 0 ; index_marker < list_markers.size() ; index_marker++)
    {
        markers mark = list_markers[index_marker];
        std::vector<stereo_match> temp_3d_points;


        for(int k = 0 ; k < mark.size() ; k++)
        {
            Vector2i val_comp = mark[k];
            for(int i = 0; i < list_projection.size(); i ++)
            {
                Vector2i val = (list_projection[i].first_point).cast<int>();
                float norm = (val_comp - val).cast<float>().norm();

                if(norm < min_l2_dist)
                {
                    if(list_projection[i].taken!=1)
                    {
                        temp_3d_points.push_back(list_projection[i]);
                        list_projection[i].taken = 1;
                    }
                }
            }
        }
        if(temp_3d_points.size() > 4)
        {
            list_objects.push_back(temp_3d_points);
            list_correct_markers.push_back(list_markers[index_marker]);
        }
    }

    if(merge_object)
    {
        get_closest_object(list_objects);
        //merge_segmented_objects(list_markers, list_projection);
    }
}



void delete_outliers_from(int surface_min_person, projected_objects_image list_3d_points_segmented_objects,
                          projected_objects_image &list_3d_points_segmented_wo_outliers, std::vector<float> &mean_z, bool verbose)
{

    //cout << "number of object " << list_3d_points_segmented_objects.size() << endl;
    for (int index_object_3d = 0 ; index_object_3d < list_3d_points_segmented_objects.size(); index_object_3d++) {

        //cout << "object number " << index_object_3d << "    " << list_3d_points_segmented_objects[index_object_3d].size() << endl;

        if(list_3d_points_segmented_objects[index_object_3d].size() > surface_min_person)
        {
            projected_object temp_vect = list_3d_points_segmented_objects[index_object_3d];
            std::sort(begin(temp_vect), end(temp_vect), ascending_depth_reprojection());
            list_3d_points_segmented_objects[index_object_3d] = temp_vect;
            VectorXf z_points(temp_vect.size());
            for(size_t i = 0 ; i < temp_vect.size() ; i++)
            {
                z_points[i] = temp_vect[i].point_projected(0);
            }
            float Q2 = z_points.mean();
            int n = z_points.rows();
            float Q1_vect = z_points[int(0.25*n)];
            float Q3_vect = z_points[int(0.75*n)];
            float IQR = Q3_vect - Q1_vect;

            float lower_bound = Q1_vect - 1.5 * IQR;
            float upper_bound = Q3_vect + 1.5 * IQR;

            VectorXf inliers(temp_vect.size());
            std::vector<int> inlier_indexes;
            int n_inliers = 0;

            //cout << "taille " << z_points.rows() << "   " << list_3d_points_segmented.size() << endl;
            projected_object vect_wo_outliers;

            for(size_t i = 0 ; i < z_points.rows() ; i++  )
            {
                if(z_points[i] > lower_bound &&  z_points[i] < upper_bound)
                {
                    inliers[n_inliers] = z_points[i];
                    inlier_indexes.push_back(i);
                    n_inliers ++;
                    //((list_3d_points_segmented[index_3d])[i])[2] = 0;
                    vect_wo_outliers.push_back((list_3d_points_segmented_objects[index_object_3d])[i]);
                }
            }

            list_3d_points_segmented_wo_outliers.push_back(vect_wo_outliers);

            inliers = inliers.segment(0, n_inliers);

            float mean_ = inliers.mean();

            mean_z[index_object_3d] = mean_;

        }

        //cout << " okay objet " << index_object_3d << endl;

    }

    //cout << "end of delete_outliers_from" << endl;


}



void save_3d_function(int nrows, int ncols, bool plane_z_mean, std::vector<markers> list_markers,
                      std::vector<float> mean_z , projected_objects_image list_objects_wo_outliers, stereo_params_cv stereo_par )
{


    MatrixXf results_3d = MatrixXf::Zero(nrows, ncols);

    if(plane_z_mean)
    {
        for(size_t index_marker = 0 ; index_marker < list_markers.size() ; index_marker++)
        {
            markers mark = list_markers[index_marker];
            float value = mean_z[index_marker];
            for(size_t i = 0; i < mark.size() ; i++)
            {
                int row = mark[i].y();
                int col = mark[i].x();
                if(value > 0)
                    results_3d(row, col) = value;
            }


            float min_x = 10000000;
            float min_y = 10000000;
            float max_x = -10000000;
            float max_y = -10000000;


            int densite = 50;


            for(int idx = 0 ; idx < list_objects_wo_outliers[index_marker].size(); idx++)
            {

                if( min_x > list_objects_wo_outliers[index_marker][idx].point_projected(2) )
                    min_x = list_objects_wo_outliers[index_marker][idx].point_projected(2) ;


                if( max_x < list_objects_wo_outliers[index_marker][idx].point_projected(2)  )
                    max_x = list_objects_wo_outliers[index_marker][idx].point_projected(2) ;


                if( min_y > list_objects_wo_outliers[index_marker][idx].point_projected(1)  )
                    min_y = list_objects_wo_outliers[index_marker][idx].point_projected(1) ;


                if( max_y < list_objects_wo_outliers[index_marker][idx].point_projected(1)  )
                    max_y = list_objects_wo_outliers[index_marker][idx].point_projected(1) ;

            }

            int cur = 0;

            std::vector<Point3f> object_points;

            for(int idx_x = int(min_x); idx_x < int(max_x); idx_x = idx_x + densite)
            {
                for(int idx_y = int(min_y); idx_y < int(max_y); idx_y = idx_y + densite)
                {
                    object_points.push_back(  Point3f(idx_x,idx_y,value) );
                    cur ++;
                }
            }

            //cout << endl << endl;

            cout << "taille 3D object " << object_points.size() << endl;

            cv::Mat rvec = stereo_par.R;
            cv::Mat tvec = stereo_par.T;
            cv::Mat camera_matrix = stereo_par.M1;
            cv::Mat distCoeffs = stereo_par.D1;
            std::vector<Point2f> imagePoints;

            cv::projectPoints(object_points,rvec,tvec,camera_matrix,distCoeffs,imagePoints);

            std::cout << "un rectangle de x " <<  min_x << " ; " << max_x << "     y " << min_y << " ; " << max_y << endl;

            int nb_okay = 0;

            std::vector<Vector3d> new_list_3d_object;

            for(size_t id_3d_pt = 0 ; id_3d_pt < imagePoints.size(); id_3d_pt++)
            {
                Vector2d _2d_projected_point(imagePoints[id_3d_pt].y, imagePoints[id_3d_pt].x);

                for(size_t id_2d_pt = 0; id_2d_pt < mark.size() ; id_2d_pt++)
                {
                    Vector2d pt_marked = mark[id_2d_pt].cast<double>();
                    float dist_ = (_2d_projected_point - pt_marked).norm();
                    if(dist_ < 2)
                    {
                        nb_okay++;
                        Vector3d _new_3d_pt(object_points[id_3d_pt].z, object_points[id_3d_pt].y, object_points[id_3d_pt].x);
                        new_list_3d_object.push_back(_new_3d_pt);
                        break;
                    }
                }
            }

            if( nb_okay >  list_objects_wo_outliers[index_marker].size())
            {
                for(size_t id_3d_pt = 0 ; id_3d_pt < new_list_3d_object.size(); id_3d_pt++)
                {


                }
            }
            else
            {
                for(int idx = 0 ; idx < list_objects_wo_outliers[index_marker].size(); idx++)
                {

                }

            }



            //cout << "ajout " << nb_okay << "  " << list_3d_points_segmented_wo_outliers[index_marker].size() << endl << endl << endl;

        }
    }
    else
    {
        for(size_t index_marker = 0 ; index_marker < list_markers.size() ; index_marker++)
        {
            markers mark = list_markers[index_marker];
            float value = mean_z[index_marker];
            for(size_t i = 0; i < mark.size() ; i++)
            {
                int row = mark[i].y();
                int col = mark[i].x();
                if(value > 0)
                    results_3d(row, col) = value;
            }



            for(int idx = 0 ; idx < list_objects_wo_outliers[index_marker].size(); idx++)
            {



            }
        }
    }

}







}
