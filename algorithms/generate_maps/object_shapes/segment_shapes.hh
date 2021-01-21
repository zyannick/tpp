#pragma once

#include "algorithms/generate_maps/sparse_disparity_maps.hh"
#include "algorithms/segmentation.hh"
#include "algorithms/fitting.hh"
#include "global_utils.hh"
#include "algorithms/kernels.hh"
#include "core.hpp"


namespace tpp {


  std::pair<MatrixXf, MatrixXf> perform_otsu(Mat left_view, Mat right_view, size_t index_img)
  {
      //cout << "perform_otsu start" << endl;
      MatrixXf dst_left, dst_right;
      //MatrixXf edges_left = mat_to_eigen(grad);
      //MatrixXf edges_left = pcr.pc_left;

      MatrixXf left_eigen = mat_to_eigen(left_view);
      int th = otsu_segmentation(left_eigen);
      //cout << "seuil " << th << endl;
      binarize_image(left_eigen,  dst_left , th , 0 , 255);
      Mat resl = eigen_to_mat(dst_left);

      imwrite("..//Segmented_images//" + to_string(index_img) + "_left.png", resl);

      MatrixXf right_eigen = mat_to_eigen(right_view);
      th = otsu_segmentation(right_eigen);
      //cout << "seuil " << th << endl;
      binarize_image(right_eigen,  dst_right , th , 0 , 255);
      Mat resr = eigen_to_mat(dst_right);

      imwrite("..//Segmented_images//" + to_string(index_img) + "_right.png", resr);
      //cout << "perform_otsu end" << endl;
      return std::pair(dst_left, dst_right);
  }


void segment_objects(int nrows, int ncols, MatrixXf dst, std::vector<markers> &list_markers, bool verbose = false)
{
    size_t index = 0;

    for(int row = 0 ; row < nrows; row++)
    {
        for(int col = 0; col < ncols; col++)
        {
            if(int(dst(row,col))==255)
            {
                markers mark;
                // take the point in format (y,x)
                Vector2i tmp_v;
                tmp_v.x() = col;
                tmp_v.y() = row;
                mark.push_back(tmp_v);
                populate_marker_segmentation(mark, row, col, dst , index);
                list_markers.push_back(mark);
                index++;
            }
        }
    }
    if (verbose)
    cout << "number of markers " << list_markers.size() << "   " << list_markers[0].size() << endl;
}




void merge_segmented_objects(std::vector<markers> list_markers, std::vector<stereo_match> list_projection)
{
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

                if(true)
                {
                    if(list_projection[i].taken!=1)
                    {
                        temp_3d_points.push_back(list_projection[i]);
                        list_projection[i].taken = 1;
                    }
                }
            }
        }
    }
}

std::pair<float, float> get_mean_disparity(projected_object object_seg)
{
    float disp_x = 0;
    float disp_y = 0;
    for(int idx = 0; idx < object_seg.size(); idx++ )
    {
        disp_y += ( object_seg[idx].first_point(0) - object_seg[idx].second_point(0) ) / (float)object_seg.size();
        disp_x += ( object_seg[idx].first_point(1) - object_seg[idx].second_point(1) ) / (float)object_seg.size();
    }

    return std::pair(disp_y, disp_x);
}



}
