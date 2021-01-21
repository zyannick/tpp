
#pragma once

#include "algorithms/generate_maps/sparse_disparity_maps.hh"
#include "algorithms/segmentation.hh"
#include "algorithms/fitting.hh"
#include "global_utils.hh"
#include "algorithms/kernels.hh"
#include "core.hpp"
#include "algorithms/generate_maps/object_shapes.hh"
#include "algorithms/generate_maps/utils_matches.hh"
#include "algorithms/generate_maps/generate_imaget.hh"
#include "algorithms/generate_maps/save_images.hh"

namespace tpp {


  void dense_disparity_map_liste_labelling_motion(base_options<float> stm)
  {



      std::vector<string> imageListCam1, imageListCam2;
      string inputFilename;

      int id_stereo_mach = 0;


      string path_to_camera1_images = string("..//").append(stm.experience_name).append("//").append(stm.experience_name).append(stm.sub_experience_name).append(string("//Camera1"));
      string path_to_camera2_images = string("..//").append(stm.experience_name).append("//").append(stm.experience_name).append(stm.sub_experience_name).append(string("//Camera2"));

      imageListCam1 = get_list_of_files(path_to_camera1_images);
      imageListCam2 = get_list_of_files(path_to_camera2_images);

      stereo_params_cv stereo_par;
      stereo_par.retreive_values();

      size_t nimages = imageListCam1.size();

      cout << "number of pairs " << nimages << endl ;
      cout << "--------------------------------------------------------------------------------------------------------------------" << endl << endl << endl;


      string dir_name = ".//3d_points//";

      create_directory(dir_name);

      namespace fs = std::experimental::filesystem;

      string label_dir = (fs::current_path()).string() + string("//..//") + string("feature_label_motion");

      create_directory(label_dir);

      string label_path = label_dir + "//" + name_from_type_of_features(stm.tof);


      create_directory(label_path);

      label_path = label_path + "//" + stm.experience_name;

      create_directory(label_path);

      label_path = label_path + "//" + stm.sub_experience_name;

      create_directory(label_path);

      projected_objects_image previous_coords;
      projected_objects_image current_coords;

      std::vector<projected_objects_image> object_in_n_frames;

      std::vector<stereo_match> live_stereo_match;

      phase_congruency_result<MatrixXf> pcr_1;
      phase_congruency_result<MatrixXf> pcr_2;

      Vector2d min_disp(0,0);

      label_path = label_path + "//_" + to_string(stm.eps) + "_";
      create_directory(label_path);

      int id_matches = 1;

      stm.verbose = false;

      for(size_t index_img = 0 ;index_img < nimages ; index_img = index_img + 2  )
      {

          cout << "-----------------------------------------------------------------------------------------------------------" << endl;
          cout << "Pair " << index_img << endl;

          //images at time t

          string path_at_this_index = string(label_path) + "//_image_" + to_string(index_img) + "_labelled";
          std::experimental::filesystem::remove_all(path_at_this_index);
          create_directory(path_at_this_index);
          string path_at_this_index_left = path_at_this_index + "//left";
          create_directory(path_at_this_index_left);
          string path_at_this_index_right = path_at_this_index + "//right";
          create_directory(path_at_this_index_right);


          std::vector<stereo_match> matchings_stereo_1;

          std::pair<Mat, Mat> pair_view_1 = format_images(int(index_img), label_path, imageListCam1, imageListCam2, stm.scale_factor);

          Mat left_view_1 =  pair_view_1.first;
          Mat right_view_1 = pair_view_1.second;

          save_ori_images(left_view_1, right_view_1, index_img, "..//motion_Oricy//", stm.experience_name, stm.sub_experience_name);


          std::pair<MatrixXf, MatrixXf> pair_otsu = perform_otsu(left_view_1, right_view_1, index_img);
          MatrixXf dst_left_1 = pair_otsu.first;
          MatrixXf dst_right_1 = pair_otsu.second;



          //cout << "after otsu" << endl;

          int ncols = left_view_1.cols;
          int nrows = left_view_1.rows;

          projected_objects_image object_images_left1_2_right1;
          projected_objects_image object_images_left2_2_right2;

          if(index_img == 0)
          {
              matching_ir_stereo_mat_pair( left_view_1 , right_view_1, pcr_1, matchings_stereo_1, stm, index_img);
          }
          else
          {
              matchings_stereo_1 = live_stereo_match;
              pcr_1 = pcr_2;
          }
          return;

          std::vector<markers> list_markers_1;


          clean_matches(ncols, nrows, min_disp, dst_left_1, matchings_stereo_1, object_images_left1_2_right1, stm.surface_min_person, list_markers_1);


          //pcr_1.print_values();

          if(index_img+1 == nimages)
              break;


          std::vector<stereo_match> matchings_stereo_2;

          std::pair<Mat, Mat> pair_view_2 = format_images(int(index_img) + 1 , label_path, imageListCam1, imageListCam2, stm.scale_factor);


          Mat left_view_2 =  pair_view_2.first;
          Mat right_view_2 = pair_view_2.second;

          std::pair<MatrixXf, MatrixXf> pair_otsu_2 = perform_otsu(left_view_2, right_view_2, index_img);
          MatrixXf dst_left_2 = pair_otsu_2.first;
          MatrixXf dst_right_2 = pair_otsu_2.second;

          MatrixXf left_eigen_1 = mat_to_eigen(left_view_1);
          MatrixXf right_eigen_1 = mat_to_eigen(right_view_1);
          MatrixXf left_eigen_2 = mat_to_eigen(left_view_2);
          MatrixXf right_eigen_2 = mat_to_eigen(right_view_2);


          matching_ir_stereo_mat_pair(left_view_2 , right_view_2, pcr_2, matchings_stereo_2, stm, index_img);
          std::vector<markers> list_markers;

          clean_matches(ncols, nrows, min_disp, dst_left_1, matchings_stereo_2, object_images_left2_2_right2, stm.surface_min_person, list_markers);

          //save_pc_images(pcr_1, index_img, "..//motion_ResultsPC//", stm.experience_name, stm.sub_experience_name);


          std::vector<stereo_match>  matchings_stereo_l1_2_l2 =  matching_stereo_ordering_constraint(phase_congruency_result<MatrixXf>(pcr_1, pcr_2, 0), stm);


          std::vector<stereo_match>  matchings_stereo_r1_2_r2 =  matching_stereo_ordering_constraint(phase_congruency_result<MatrixXf>(pcr_1, pcr_2, 1), stm);



          /*cout << "m1 " << matchings_stereo_1.size() << endl;
          cout << "m5 " << matchings_stereo_2.size() << endl;
          cout << "m2 " << matchings_stereo_l1_2_l2.size() << endl;
          cout << "m3 " << matchings_stereo_r1_2_r2.size() << endl;*/


          std::sort(matchings_stereo_1.begin(), matchings_stereo_1.end() , [&](stereo_match& a, stereo_match& b){return a.first_point.norm() > b.first_point.norm();});

          std::sort(matchings_stereo_2.begin(), matchings_stereo_2.end() , [&](stereo_match& a, stereo_match& b){return a.first_point.norm() > b.first_point.norm();});

          std::sort(matchings_stereo_l1_2_l2.begin(), matchings_stereo_l1_2_l2.end() , [&](stereo_match& a, stereo_match& b){return a.first_point.norm() > b.first_point.norm();});

          std::sort(matchings_stereo_r1_2_r2.begin(), matchings_stereo_r1_2_r2.end() , [&](stereo_match& a, stereo_match& b){return a.first_point.norm() > b.first_point.norm();});




          std::vector<stereo_match> consistency_matches;

          for(int idm = 0 ; idm <  matchings_stereo_1.size(); idm ++)
          {
              stereo_match m1 = matchings_stereo_1[idm];
              stereo_match m2;
              stereo_match m3;
              for(size_t l1 = 0; l1 < matchings_stereo_l1_2_l2.size(); l1++)
              {
                  stereo_match l1l2 =  matchings_stereo_l1_2_l2[l1];
                  if( (m1.first_point - l1l2.first_point).norm() < 2 &&  l1l2.taken_time_consistency == -1)
                  {
                      l1l2.taken_time_consistency = 1;
                      m2 = l1l2;
                      break;
                  }
              }

              for(size_t r1 = 0; r1 < matchings_stereo_r1_2_r2.size(); r1++)
              {
                  stereo_match r1r2 =  matchings_stereo_r1_2_r2[r1];
                  if( (m1.second_point - r1r2.first_point).norm() < 2 && r1r2.taken_time_consistency == -1)
                  {
                      r1r2.taken_time_consistency = 1;
                      m3 = r1r2;
                      break;
                  }
              }

              if(m2.taken_time_consistency == 1 && m3.taken_time_consistency == 1)
              {
                  stereo_match st(m2.second_point, m3.second_point, m1);
                  consistency_matches.push_back(st);
              }
          }


          //cout << "m4 " << consistency_matches.size() << endl << endl;


          //consistency_matches.sort( [&](stereo_match& a, stereo_match& b){return a.first_point.norm() > b.first_point.norm();});

          std::vector<stereo_match> robust_time_consistency_matches;

          for(size_t l2 = 0; l2 < matchings_stereo_2.size(); l2++)
          {
              stereo_match m4;
              for(size_t c = 0; c < consistency_matches.size(); c++)
              {
                  stereo_match cm = consistency_matches[c];
                  if( (matchings_stereo_2[l2].first_point - cm.first_point).norm() < 2
                          && (matchings_stereo_2[l2].second_point - cm.second_point).norm() < 2
                          && cm.taken_time_consistency == -1
                          && matchings_stereo_2[l2].taken_time_consistency == -1)
                  {
                      cm.taken_time_consistency = 1;
                      matchings_stereo_2[l2].taken_time_consistency = 1;
                      m4 = cm;
                      break;
                  }
              }
              if(m4.taken_time_consistency == 1)
              {
                  robust_time_consistency_matches.push_back(m4);
              }
          }

          save_stereo_segmentation_results(dst_left_2, dst_right_2,  stm.eps,robust_time_consistency_matches, index_img, "..//motion_Segmentation_results//", stm.experience_name, stm.sub_experience_name);

          //cout << "m6 before adding " << robust_time_consistency_matches.size() << endl;

          for(size_t l2 = 0; l2 < matchings_stereo_2.size(); l2++)
          {
              if(matchings_stereo_2[l2].taken_time_consistency == -1)
              {
                  robust_time_consistency_matches.push_back(matchings_stereo_2[l2]);
              }
          }




          cout << "list of stereo matches " << robust_time_consistency_matches.size() << endl;

          disparity_3d_projection(robust_time_consistency_matches);

          cout << "list of stereo matches " << robust_time_consistency_matches.size() << endl;


          from_matches_to_objects(robust_time_consistency_matches, object_images_left2_2_right2, id_matches);

          cout << "list of stereo matches " << robust_time_consistency_matches.size() << endl;


          save_image_from_z(object_images_left2_2_right2, left_eigen_2, nrows, ncols,index_img ,
                            get_string_from_tof_enum(stm.tof), 0.1, "..//motion_Results//", stm.experience_name, stm.sub_experience_name, stm.eps);

          cout << "list of stereo matches " << robust_time_consistency_matches.size() << endl;

          save_stereo_images_markers(list_markers, robust_time_consistency_matches,  nrows, ncols, index_img, false, "..//motion_Matching_on//", stm.experience_name, stm.sub_experience_name);

          for(size_t idx = 0; idx < robust_time_consistency_matches.size(); idx++)
          {
              //cout << "Profondeur " << robust_time_consistency_matches[idx].depth << "  " <<  robust_time_consistency_matches[idx].id << endl;
          }

          cout << "list of stereo matches " << robust_time_consistency_matches.size() << endl;


          generate_view_from_list_of_two_points_motion<float>(object_images_left2_2_right2, nrows, ncols, path_at_this_index
                                                       ,  path_at_this_index_left, path_at_this_index_right, index_img, stm.verbose);

          cout << "list of stereo matches " << robust_time_consistency_matches.size() << endl;

          live_stereo_match = robust_time_consistency_matches;


          cout << endl << endl;


      }
  }

}
