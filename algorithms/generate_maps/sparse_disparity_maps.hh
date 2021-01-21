#pragma once


#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>

#include "core.hpp"
#include "algorithms/matching.hh"
#include "algorithms/morphology.hh"
#include "algorithms/segmentation.hh"
#include "algorithms/miscellaneous.hh"
#include "algorithms/interpolation.hh"
#include "calibration_and_3d/recontruction3D/recontruction3D.hh"
#include "utils.hh"


#include <omp.h>

using namespace std;
using namespace Eigen;



namespace tpp
{



template<typename T>
void disparity_3d_projection(std::vector<stereo_match> matchings_stereo, std::vector<Matrix<T,3,1>> &list_points, bool is_rectified = true )
{
    stereo_params_cv stereo_par;
    stereo_par.retreive_values();

    //cout << "disparity_3d_projection start" << endl;

    //cout << stereo_par.F << endl;


    if(is_rectified)
    {
        for(stereo_match st: matchings_stereo)
        {
            Vector2d lp = st.first_point;
            Vector2d rp = st.second_point;

            //cout << lp.y() << "  "  << lp.x()  << "             " << rp.y()  << "   "  << rp.y()  << endl;


            Mat points3D_;
            triangulate_point_to_3D_corrected_match(stereo_par.F, stereo_par.M1, stereo_par.M2, stereo_par.P1, stereo_par.P2,
                                                    stereo_par.R1, stereo_par.R2, stereo_par.T, stereo_par.D1, stereo_par.D2,
                                                         Point2f(lp(1),lp(0)),
                                                         Point2f(rp(1),rp(0)), points3D_);

            //cout << points3D_.at<double>(0, 0) << "  " << points3D_.at<double>(1, 0) << "  " << points3D_.at<double>(2, 0) << "   " << points3D_.at<double>(4, 0) <<   endl;
            //list_points.push_back( Matrix<T,3,1>( points3D_.at<double>(2, 0) ,  points3D_.at<double>(1, 0) ,  points3D_.at<double>(0, 0)   ));
            list_points.push_back( Matrix<T,3,1>( points3D_.at<double>(0, 0) ,  points3D_.at<double>(1, 0) ,  points3D_.at<double>(2, 0)   ));
            st.point_projected.x() = points3D_.at<double>(0, 0);
            st.point_projected.y() = points3D_.at<double>(1, 0);
            st.point_projected.z() = points3D_.at<double>(2, 0);
        }
    }

    //cout << "disparity_3d_projection end" << endl;
    return ;
}


void disparity_3d_projection(std::vector<stereo_match> &matchings_stereo, bool is_rectified = true )
{
    stereo_params_cv stereo_par;
    stereo_par.retreive_values();

    //cout << "disparity_3d_projection start" << endl;

    //cout << stereo_par.F << endl;


    //if(is_rectified)
    {
        for(stereo_match &st: matchings_stereo)
        {
            Vector2d lp = st.first_point;
            Vector2d rp = st.second_point;

            //cout << lp.y() << "  "  << lp.x()  << "---" << rp.y()  << "   "  << rp.y()  << endl;

            Mat points3D_;
            triangulate_point_to_3D_corrected_match(stereo_par.F, stereo_par.M1, stereo_par.M2, stereo_par.P1, stereo_par.P2,
                                                    stereo_par.R1, stereo_par.R2, stereo_par.T, stereo_par.D1, stereo_par.D2,
                                                         Point2f(lp.x(),lp.y()),
                                                         Point2f(rp.x(),rp.y()), points3D_);

            //cout << points3D_.at<double>(0, 0) << "  " << points3D_.at<double>(1, 0) << "  " << points3D_.at<double>(2, 0) << "   " << points3D_.at<double>(4, 0) <<   endl;
            //st.point_projected = Vector3d( points3D_.at<double>(2, 0) ,  points3D_.at<double>(1, 0) ,  points3D_.at<double>(0, 0) );
            st.point_projected.x() = points3D_.at<double>(0, 0);
            st.point_projected.y() = points3D_.at<double>(1, 0);
            st.point_projected.z() = points3D_.at<double>(2, 0);
        }
    }

    //cout << "disparity_3d_projection end" << endl;
    return ;
}



void disparity_map_liste(int wind_row, int wind_col ,bool save = true,
                         bool rect = true,MATCHING_SIMILARITY_METHOD dist_method = MATCHING_SIMILARITY_METHOD::LADES_SIMILARITY,
                         SUB_PIXEL_MATCHING_PRECISE sub_pixel = SUB_PIXEL_MATCHING_PRECISE::PHASE_CORRELATION_FORROSH)
{



    std::vector<string> imageListCam1, imageListCam2;
    string inputFilename;
    imageListCam1 = get_list_of_files("..\\ImagesTest\\Camera1");
    imageListCam2 = get_list_of_files("..\\ImagesTest\\Camera2");

    stereo_params_cv stereo_par;
    stereo_par.retreive_values();

    float eps = float(0.1);

    size_t nimages = imageListCam1.size();

    cout << "number of pairs " << nimages << endl ;
    cout << "-------------------------------------------------------------------------" << endl << endl << endl;

    std::vector<stereo_match> matchings_stereo;
    phase_congruency_result<MatrixXf> pcr;




    for(size_t i = 0 ;i < nimages ; i++)
    {
        std::vector<Vector3d> list_3d_points;

        Mat left_view_16 = imread(imageListCam1[i], CV_16U);
        Mat left_view;

        double minVal, maxVal;
        Point minLoc, maxLoc;
        minMaxLoc(left_view_16, &minVal, &maxVal, &minLoc, &maxLoc);
        left_view_16.convertTo(left_view, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));

        Mat right_view_16 = imread(imageListCam2[i], CV_16U);
        Mat right_view;

        minMaxLoc(right_view_16, &minVal, &maxVal, &minLoc, &maxLoc);
        right_view_16.convertTo(right_view, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));

        cout << "disparity_map_pairs" << endl;
        //disparity_map_pairs(pcr, matchings_stereo, left_view , right_view, stereo_par, wind_row, wind_col, eps, rect, int(i) ,0, dist_method,sub_pixel );

        cout << "disparity_3d_projection" << endl;
        disparity_3d_projection(matchings_stereo, list_3d_points);

        const int nrows = left_view_16.rows;
        const int ncols = left_view_16.cols;
        Mat result_3d = cv::Mat::zeros(cv::Size(ncols, nrows), CV_64F);
        MatrixXf results_3d = MatrixXf::Zero(nrows, ncols);

        size_t j = 0;

        std::vector<Vector2d> list_points;

        float val_min = 100000000;
        float val_max = 0;

        for(stereo_match mt:  matchings_stereo)
        {
            int x = int(mt.first_point(1));
            int y = int(mt.first_point(0));
            float val_z = float(list_3d_points[j](2));
            j++;
            if(val_z < 0)
                continue;

            results_3d(y,x) = val_z;

            if(val_max < val_z)
            {
                val_max = val_z;
            }

            if(val_min < val_z)
            {
                val_min = val_z;
            }

            list_points.push_back(Vector2d(y,x));
        }

        normalize_matrix(results_3d, 1);

        Matrix<Vector3d, Dynamic, Dynamic> colored_img(nrows, ncols);

        Matrix<Vector3d, Dynamic, Dynamic> blent_img(nrows, ncols);

        MatrixXf img_left_tir = mat_to_eigen(left_view);
        normalize_matrix(img_left_tir,  255);

        float alpha = 0.5;
        float beta;

        beta = ( 1.0 - alpha );

        for(int row = 0 ; row < nrows; row++)
        {
            for(int col = 0; col < ncols ; col ++)
            {
                float r, g , b;
                colormap<float>( ColorMapType::COLOR_MAP_TYPE_VIRIDIS, results_3d(row , col), r, g, b  );
                colored_img(row,col) = 255.0 * Vector3d(r, g, b);
                float new_red = alpha * img_left_tir(row, col) + beta * 255 * r;
                float new_green = alpha * img_left_tir(row, col) + beta * 255 * g;
                float new_blue = alpha * img_left_tir(row, col) + beta * 255 * b;

                blent_img(row, col) = Vector3d(new_red, new_green, new_blue);
            }
        }

        Mat result_opencv = eigen_to_mat_template<Vector3d>(blent_img);
        cout << "here 2" << endl;

        string file_name = string("..//Results//");
        file_name.append(string("_")).append(to_string(i)).append(string(".png"));

        imwrite(file_name, result_opencv);


        matchings_stereo.clear();

    }
}



void test_disparity_map_liste(int wind_row, int wind_col ,bool save = true,
                               bool rect = true,MATCHING_SIMILARITY_METHOD dist_method = MATCHING_SIMILARITY_METHOD::LADES_SIMILARITY,
                               SUB_PIXEL_MATCHING_PRECISE sub_pixel = SUB_PIXEL_MATCHING_PRECISE::PHASE_CORRELATION_NAGASHIMA)
{
    std::vector<string> left_images;
    string inputFilename = miscellaneous;

    left_images = get_list_of_files("..\\grande");
    
    int nimages = left_images.size();

    stereo_params_cv stereo_par;
    stereo_par.retreive_values();
    
    
    float eps = 0.1;

    for (int ster = 0; ster < nimages; ster++)
    {

        cout << "image numero " << ster << endl;

        float offset = 20;

        while(offset <= 30)
        {
            
            cout << "offset " << offset << endl;
            
            string left_image =left_images[ster];
            Mat left_view_16 = imread(left_image, CV_16U);
            Mat left_view;

            double minVal, maxVal;
            Point minLoc, maxLoc;
            minMaxLoc(left_view_16, &minVal, &maxVal, &minLoc, &maxLoc);
            left_view_16.convertTo(left_view, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));

            int nrows = left_view.rows;
            int ncols = left_view.cols;

            int po = int(offset*100);

            Mat map_matrix = Mat::zeros(2,3,CV_32FC1);

            map_matrix.at<float>(0,0) = 1;
            map_matrix.at<float>(1,1) = 1;
            map_matrix.at<float>(0,2) = 0;
            map_matrix.at<float>(0,0) = 0;

            Point2f srcTri[3];
            Point2f dstTri[3];
            Mat warp_mat( 2, 3, CV_32FC1 );

            /// Set your 3 points to calculate the  Affine Transform
            srcTri[0] = Point2f( 0,0 );
            srcTri[1] = Point2f( 30, 30 );
            srcTri[2] = Point2f( 50,56  );

            dstTri[0] = Point2f( 0 + offset, 0 );
            dstTri[1] = Point2f( 30 + offset, 30 );
            dstTri[2] = Point2f( 50 + offset, 56 );

            /// Get the Affine Transform
            warp_mat = getAffineTransform( srcTri, dstTri );

            Mat dst  = Mat::zeros( left_view.rows, left_view.cols, left_view.type() );;

            /// Apply the Affine Transform just found to the src image
            warpAffine( left_view, dst, warp_mat, dst.size() ,
                        INTER_LANCZOS4,
                        BORDER_REPLICATE);

            if(sub_pixel !=  SUB_PIXEL_MATCHING_PRECISE::INTERPOLATION)
            {
                //disparity_map_pairs(left_view , dst, stereo_par, wind_row, wind_col, eps, rect,ster,offset, dist_method,sub_pixel );
            }
            else
            {
                int factor = 4;
                Mat src_new, dst_new;
                src_new = Mat::zeros(factor*left_view.rows, factor*left_view.cols, left_view.type() );
                dst_new = Mat::zeros(factor*dst.rows, factor*dst.cols, dst.type() );
                resize(left_view, src_new, src_new.size(), 0, 0, INTER_CUBIC );
                resize(dst, dst_new, dst_new.size(), 0, 0, INTER_CUBIC );
                //disparity_map_pairs(left_view , dst, stereo_par, wind_row, wind_col, eps, rect,ster,offset, dist_method,sub_pixel );
            }
            offset = offset + 0.125;
        }
        offset = 0;
    }
    return;
}



}
