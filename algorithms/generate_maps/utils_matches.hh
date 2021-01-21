#pragma once

#include "algorithms/generate_maps/sparse_disparity_maps.hh"
#include "algorithms/segmentation.hh"
#include "algorithms/fitting.hh"
#include "global_utils.hh"
#include "algorithms/kernels.hh"
#include "object_shapes.hh"
#include "core.hpp"


namespace tpp {


  template <typename T>
  void list_reprojection_save(std::vector<stereo_match> &list_projection, std::vector<stereo_match> matchings_stereo, std::vector<Vector3d> list_3d_points)
  {
      size_t index_matche = 0;

      //cout << "list_reprojection_save start" << endl;

      for(stereo_match mt:  matchings_stereo)
      {
          //cout << "lowe_ratio " << mt.lowe_ratio << endl;
          int x = int(mt.first_point.x());
          int y = int(mt.first_point.y());
          //if( list_3d_points[index_matche](2)  > 0 && list_3d_points[index_matche](2) < 10000.0)

          {
              list_projection.push_back(stereo_match(mt.first_point, mt.second_point, list_3d_points[index_matche]));
          }
          index_matche++;
      }

      //cout << "list_reprojection_save end" << endl;
  }


  int match_in_list_of_matches(Vector2d first_point, Vector2d second_point, std::vector<stereo_match> matchings_stereo)
  {
      for(int i = 0; i < matchings_stereo.size(); i++)
      {
          if( (matchings_stereo[i].first_point - first_point).norm() == 0
                  &&  (matchings_stereo[i].second_point - second_point).norm() == 0 )
          {
              return i;
          }

      }
      return -1;
  }

  void from_objects_to_matches(std::vector<stereo_match> &matchings_stereo_1, projected_objects_image &lowo)
  {
      std::vector<stereo_match> matchings_stereo_clean;

      for(size_t i = 0; i < lowo.size(); i++)
      {
          for(size_t j = 0; j < lowo[i].size(); j++)
          {
              int idx = match_in_list_of_matches(lowo[i][j].first_point, lowo[i][j].second_point, matchings_stereo_1);
              if( idx >= 0 )
              {
                  lowo[i][j].list_first_view_matches = matchings_stereo_1[idx].list_first_view_matches;
                  lowo[i][j].list_second_view_matches = matchings_stereo_1[idx].list_second_view_matches;
                  matchings_stereo_clean.push_back(matchings_stereo_1[idx]);
              }
          }
      }
      matchings_stereo_1 = matchings_stereo_clean;
  }

  void from_matches_to_objects(std::vector<stereo_match> &matchings_stereo_1, projected_objects_image &lowo, int &id_match)
  {
      projected_objects_image lowo_up;
      for(size_t i = 0; i < lowo.size(); i++)
      {
          projected_object po_temp;
          for(size_t j = 0; j < lowo[i].size(); j++)
          {
              int idx = match_in_list_of_matches(lowo[i][j].first_point, lowo[i][j].second_point, matchings_stereo_1);
              if( idx >= 0 )
              {
                  if(matchings_stereo_1[idx].id == -1)
                  {
                      matchings_stereo_1[idx].id = id_match++;
                      matchings_stereo_1[idx].object_number = i;
                  }
                  matchings_stereo_1[idx].initialize_taken_values();
                  po_temp.push_back(matchings_stereo_1[idx]);
              }
          }
          lowo_up.push_back(po_temp);
      }
      lowo = lowo_up;
  }




  void clean_matches(int ncols, int nrows, Vector2d min_disp, MatrixXf dst_left,
                     std::vector<stereo_match> &matchings_stereo_1, projected_objects_image &lowo,
                     int surface_min_person, std::vector<markers> &list_markers,bool verbose = true, bool del_outliers=false )
  {
      std::vector<Vector3d> list_3d_points;
      disparity_3d_projection(matchings_stereo_1, list_3d_points);
      std::vector<stereo_match> list_projection;
      list_reprojection_save<float>(list_projection, matchings_stereo_1, list_3d_points);
      segment_objects(nrows, ncols, dst_left, list_markers);
      std::vector<markers> list_correct_markers;
      projected_objects_image list_objects;


      if(verbose)
      {
          cout << "taille list_projection " << list_projection.size() << endl;
      }
      int nb_pixels = 0;
      min_disp = get_objects(list_markers, list_projection, surface_min_person, list_objects, list_correct_markers, nb_pixels,  min_disp);

      list_markers = list_correct_markers;


      std::vector<float> mean_z(list_objects.size(), 0);



      if(del_outliers)
      {
          delete_outliers_from(surface_min_person, list_objects, lowo, mean_z, verbose);
      }
      else
      {
          lowo = list_objects;
      }

      from_objects_to_matches(matchings_stereo_1, lowo);


  }



}
