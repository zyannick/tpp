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
#include "algorithms/matching.hh"
#include <boost/algorithm/string.hpp>

namespace tpp {

#ifdef vpp
using namespace vpp;
#endif


void example_infrared_and_physics()
{
    std::string tm =  "list_matches.txt";
    ofstream log_matches (tm,  std::ios_base::app);

    std::vector<stereo_match> matchings_stereo;

    int inter = 10;

    int y_l = 44;
    int x_l = 56;

    int y_r = 45;
    int x_r = 53;
    for (int i = -inter; i <= inter ; i++)
    {
        for (int j = -inter; j <= inter ; j++)
        {
            stereo_match mt(Vector2d(y_l, x_l), Vector2d(y_r +j/100.0, x_r+i/100.0),0);
            cout << y_r +j/100.0 << "   " << x_r+i/100.0 << endl;
            matchings_stereo.push_back(mt);
        }
    }


    std::vector<Vector3d> list_3d_points;
    disparity_3d_projection(matchings_stereo, list_3d_points);

    for(int icp = 0 ; icp < matchings_stereo.size(); icp ++)
    {
        if(list_3d_points[icp][2]  > 0)
        {
            log_matches << std::fixed << std::setprecision(5)  << matchings_stereo[icp].first_point.y() << ";"
                        << matchings_stereo[icp].first_point.x() << ";" << matchings_stereo[icp].second_point.y() << ";" << matchings_stereo[icp].second_point.x() << ";"
                        << list_3d_points[icp].y() << ";" << list_3d_points[icp].x() << ";" << list_3d_points[icp].z()  << endl;
        }
    }


}



void detect_decrescendo(VectorXd &U_n, VectorXd &temp_values_u_m, double m_n, long nindex, long index_img)
{
    /*temp_values_u_m(index_img) = (nb_pixels_here - mu_zero - delta_m/2 );
    U_n(index_img) = temp_values_u_m.sum();
    m_n = (U_n.segment(0, long(index_img+1))).minCoeff();*/
}

Vector2d compute_gravity_center_matches(projected_objects_image list_objects_wo_outliers)
{
    Vector2d temp2d = Vector2d::Zero();
    int nb = 0;
    for (int i = 0 ;  i < list_objects_wo_outliers.size(); i++)
    {
        for(int j = 0; j < list_objects_wo_outliers[i].size(); j++ )
        {
            temp2d(0) += list_objects_wo_outliers[i][j].first_point(0);
            temp2d(1) += list_objects_wo_outliers[i][j].first_point(1);
            nb ++;
        }
    }
    if (nb == 0)
        nb = 1;

    return temp2d/nb;
}



int get_where_to_update_disp_history(MatrixXd gravity_center_person)
{
    //cout << "get_where_to_update_disp_history " << endl;
    int wm = 0;
    if( gravity_center_person(0, 2)  == gravity_center_person(1, 2) && gravity_center_person(0, 2)  == gravity_center_person(2, 2))
    {
        wm = 0;
    }
    else
    {
        MatrixXf::Index min_index;
        gravity_center_person.col(2).minCoeff(&min_index);
        wm = int(min_index);
    }
    //cout << "get_where_to_update_disp_history end " << wm << endl;
    return wm;

}

bool correct_occlusions(double nb_pixels_here, double mu_zero,  double delta_m, double lambda, VectorXd &temp_values_u_m,
                        double &nb_total_pixels_normal, int &counting_images,
                        VectorXd &U_n,  size_t &index_img, int &retour_occlusion)
{
    cout << "voyons voir ici " << counting_images << endl;
    nb_total_pixels_normal += nb_pixels_here;

    bool occlusion;


    mu_zero = nb_total_pixels_normal / (counting_images + 1);

    temp_values_u_m(counting_images) = (nb_pixels_here - mu_zero - delta_m/2 );
    U_n(counting_images) = temp_values_u_m.sum();
    double m_n = (U_n.segment(0, counting_images +1)).minCoeff();

    double detect_nothing = m_n - U_n(counting_images);


    if(fabs(detect_nothing) < lambda)
    {
        cout << "No occlusion " << nb_pixels_here << "   " << mu_zero <<  "  "
             << detect_nothing << endl;
        occlusion = false;
    }
    else
    {
        cout << "Occlusion " << nb_pixels_here << "   " << mu_zero  <<  "  "
             << detect_nothing << endl;
        occlusion = true;
        nb_total_pixels_normal -= nb_pixels_here;
        counting_images--;
    }

    counting_images++;

    /*if(retour_occlusion == 2)
    {
        retour_occlusion = 0;
        nb_total_pixels_normal -= nb_pixels_here;
        counting_images--;

    }*/

    if(occlusion && retour_occlusion==0)
    {
        index_img--;
        retour_occlusion = 1;
    }

    return occlusion;


}

double compute_mean_distance_between_markers(markers list1, markers list2)
{
    double min_dist = 1000;
    for(size_t i = 0; i < list1.size(); i++ )
    {
        for(size_t j = 0; j < list2.size(); j++)
        {
            if( (list1[i] - list2[j] ).norm() < min_dist)
            {
                min_dist = (list1[i] - list2[j] ).norm();
            }
        }
    }
    return min_dist;
}

double compute_mean_distance_between_list_markers(std::vector<markers> list1, std::vector<markers> list2)
{
    double min_dist = 1000;
    for(size_t i = 0; i < list1.size(); i++ )
    {
        for(size_t j = 0; j < list2.size(); j++)
        {
            double val = compute_mean_distance_between_markers(list1[i], list2[j]);
            if(  val < min_dist )
            {
                min_dist = val;
            }
        }
    }
    return  min_dist;
}


Vector2d compute_gravity_center_markers(std::vector<markers> list1)
{
    Vector2d temp2d = Vector2d::Zero();
    int nb = 0;
    for (int i = 0 ;  i < list1.size(); i++)
    {
        for(int j = 0; j < list1[i].size(); j++ )
        {
            temp2d += list1[i][j].cast<double>();
            nb ++;
        }
    }
    if (nb == 0)
        nb = 1;

    return temp2d/nb;
}

class CSVReader
{
    std::string fileName;
    std::string delimeter;

public:
    CSVReader(std::string filename, std::string delm = ",") :
        fileName(filename), delimeter(delm)
    { }

    // Function to fetch data from a CSV File
    std::vector<std::vector<std::string> > getData();
};

/*
* Parses through csv file line by line and returns the data
* in vector of vector of strings.
*/
std::vector<std::vector<std::string> > CSVReader::getData()
{

    std::ifstream file(fileName);

    cout << fileName << endl;

    std::vector<std::vector<std::string> > dataList;

    std::string line = "";
    // Iterate through each line and split the content using delimeter
    while (getline(file, line))
    {
        std::vector<std::string> vec;
        boost::algorithm::split(vec, line, boost::is_any_of(delimeter));
        dataList.push_back(vec);
    }
    // Close the File
    file.close();

    return dataList;
}

bool check_image(std::vector<std::vector<std::string>> data, int index_img, bool &label_image )
{
    //cout << data.size() << endl;
    for(int i = 0; i < data.size(); i++)
    {
        int num_img = std::stoi( data[i][0] );
        if(num_img == index_img)
        {
            int label = std::stoi( data[i][1] );
            if(label == 1)
                label_image = true;
            else if (label == 0)
                label_image = false;
            return (label != -1);
        }
    }
    return false;
}

template<typename Type>
void dense_disparity_map_liste_labelling( base_options<Type> stm)
{


    cout << " dense_disparity_map_liste_labelling "  << stm.verbose << endl;
    CSVReader csvr = CSVReader(stm.sub_experience_name + ".csv", ";");
    auto data_here = csvr.getData();
    std::vector<string> imageListCam1, imageListCam2;
    string inputFilename;

    std::vector<string> imageListCam1_ref, imageListCam2_ref;


    string path_to_ref_camera1_images = string("..//").append(stm.experience_name).append("//").append(stm.experience_name).append("Ref").append(string("//Camera1"));
    string path_to_ref_camera2_images = string("..//").append(stm.experience_name).append("//").append(stm.experience_name).append("Ref").append(string("//Camera2"));

    double mean_max_val = 0.0;

    if(stm.with_map_ref)
    {
        imageListCam1_ref = get_list_of_files(path_to_ref_camera1_images);
        imageListCam2_ref = get_list_of_files(path_to_ref_camera2_images);

        Mat left_view_16_ref;
        Mat left_view_ref;

        Mat right_view_16_ref;
        Mat right_view_ref;

        size_t nimages_ref = imageListCam1_ref.size();

        for(size_t index_img = 0 ;index_img < nimages_ref ; index_img++)
        {


            left_view_16_ref = imread(imageListCam1_ref[index_img], CV_16U);
            double minVal, maxVal;
            Point minLoc, maxLoc;
            minMaxLoc(left_view_16_ref, &minVal, &maxVal, &minLoc, &maxLoc);
            mean_max_val += maxVal;
            left_view_16_ref.convertTo(left_view_ref, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));

            right_view_16_ref = imread(imageListCam2_ref[index_img], CV_16U);
            minMaxLoc(right_view_16_ref, &minVal, &maxVal, &minLoc, &maxLoc);
            mean_max_val += maxVal;
            right_view_16_ref.convertTo(right_view_ref, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));

        }

        mean_max_val = mean_max_val / (2 * nimages_ref);
    }

    //cout << "mean_max_val  " << mean_max_val << endl;



    string path_to_camera1_images = string("..//").append(stm.experience_name).append("//").append(stm.sub_experience_name).append(string("//Camera1"));
    string path_to_camera2_images = string("..//").append(stm.experience_name).append("//").append(stm.sub_experience_name).append(string("//Camera2"));

    imageListCam1 = get_list_of_files(path_to_camera1_images);
    imageListCam2 = get_list_of_files(path_to_camera2_images);

    stereo_params_cv stereo_par;
    stereo_par.retreive_values();

    size_t nimages = imageListCam1.size();

    cout << "number of pairs " << nimages << endl ;
    cout << "-------------------------------------------------------------------------" << endl << endl << endl;

    std::vector<stereo_match> matchings_stereo;
    phase_congruency_result<MatrixXf> pcr;

    string dir_name = ".//3d_points//";

    create_directory(dir_name);

    namespace fs = std::experimental::filesystem;

    string label_dir = (fs::current_path()).string() + string("//..//") + string("feature_label");

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

    std::vector<object_segmented> transient_objects;

    Vector2d min_disp = Vector2d::Zero();

    label_path = label_path + "//_" + to_string(stm.eps) + "_";
    create_directory(label_path);



    double u_zero = 0;
    double mu_zero = 0;
    double m_n = 0;
    double M_n = 0;
    double delta_m = 1000;
    double lambda = 1500;

    double nb_total_pixels_normal = 0.0;

    long nindex = long(nimages);

    VectorXd mu_n = VectorXd::Zero(nindex);

    VectorXd U_n =  VectorXd::Zero(nindex);
    VectorXd temp_values_u_m =  VectorXd::Zero(nindex);


    VectorXd T_n =  VectorXd::Zero(nindex);
    VectorXd temp_values_t_m =  VectorXd::Zero(nindex);

    VectorXd vector_t =  VectorXd::Zero(nindex);

    long start_occ = 0;
    long end_occ = 0;

    bool occlusion = false;

    int counting_images = 0;

    MatrixXd gravity_center_person = MatrixXd::Zero(3,2);

    std::string tm =   "..//" + stm.sub_experience_name + "_"  + "_nb_otsu.csv";
    ofstream log_otsu (tm);

    int retour_occlusion = 0;
    MatrixXf saved_otsu;
    std::vector<std::vector<markers>> history_markers(3);
    std::vector<double> list_number_pixels(3, 0);
    int occlusion_since = 0;

    string ok_root = "./new_root";
    create_directory(ok_root);
    create_directory(ok_root + "//" + stm.experience_name);
    create_directory(ok_root + "//" + stm.experience_name + "//" + stm.sub_experience_name);
    create_directory(ok_root + "//" + stm.experience_name + "//" + stm.sub_experience_name + "//Camera1" );
    create_directory(ok_root + "//" + stm.experience_name + "//" + stm.sub_experience_name + "//Camera2");

    string new_where_to_save = ok_root + "//" + stm.experience_name + "//" + stm.sub_experience_name;

    stm.verbose = true;

    for(size_t index_img = 0 ;index_img < nimages ; index_img++)
    {

        bool label_image = false;

        /*if(!check_image(data_here, index_img, label_image))
        {
            continue;
        }*/

        if(stm.verbose)
        {
            cout << endl << endl << endl;
            cout << "--------------------------------------------------------------------------------------------------------------------------------------" << endl;
            cout << "Pair " << index_img << endl;
        }

        string path_at_this_index = string(label_path) + "//_image_" + to_string(index_img) + "_labelled";
        std::experimental::filesystem::remove_all(path_at_this_index);
        create_directory(path_at_this_index);
        string path_at_this_index_left = path_at_this_index + "//left";
        create_directory(path_at_this_index_left);
        string path_at_this_index_right = path_at_this_index + "//right";
        create_directory(path_at_this_index_right);


        std::vector<Vector3d> list_3d_points;

        auto read_mode = CV_16U;

        //cout << "path tp image " << imageListCam1[index_img] << endl << endl;

        Mat left_view_16 = imread(imageListCam1[index_img], read_mode);
        Mat right_view_16 = imread(imageListCam2[index_img], read_mode);


        Mat left_view;

        double minVal, maxVal;
        Point minLoc, maxLoc;
        if (!stm.keep_values)
        {
            minMaxLoc(left_view_16, &minVal, &maxVal, &minLoc, &maxLoc);
        }
        else
        {
            cout << "fixed values " << endl;
            minVal = 7900;
            maxVal = 9000;
        }


        left_view_16.convertTo(left_view, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));

        if( stm.scale_factor > 1)
            cv::resize(left_view, left_view, cv::Size(), stm.scale_factor, stm.scale_factor , INTER_CUBIC );


        Mat right_view;



        if(left_view.empty())
        {
            cout << imageListCam1[index_img] << "  "  << index_img << "   "  << nimages << endl;
            continue;
        }



        if (! stm.keep_values)
        {
            minMaxLoc(right_view_16, &minVal, &maxVal, &minLoc, &maxLoc);
        }
        else
        {
            minVal = 7900;
            maxVal = 9000;
        }

        right_view_16.convertTo(right_view, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));

        imwrite(new_where_to_save +  "//Camera1//" + std::to_string(100000 + index_img) +  ".png", left_view_16 );
        imwrite(new_where_to_save +  "//Camera2//" + std::to_string(100000 + index_img) +  ".png", right_view_16 );

        if( stm.scale_factor > 1)
            cv::resize(right_view, right_view, cv::Size(), stm.scale_factor, stm.scale_factor , INTER_CUBIC );

        MatrixXf dst_left;
        MatrixXf dst_right;
        MatrixXf dst_left_dilated;
        MatrixXf dst_right_dilated;
        Mat left_temp_otsu;
        Mat right_temp_otsu;

        int non_zero_left = 0;
        int non_zero_right = 0;


        {
            cv::threshold(left_view, left_temp_otsu, 0, 255, cv::THRESH_OTSU);
            cv::threshold(right_view, right_temp_otsu, 0, 255, cv::THRESH_OTSU);

            Mat bin_left;
            cv::threshold(left_view, bin_left, 50, 255, cv::THRESH_BINARY);
            non_zero_left = count_number_of_no_zero_pixels(bin_left);
            cout << "left number " << non_zero_left << endl;
            Mat bin_right;
            cv::threshold(right_view, bin_right, 50, 255, cv::THRESH_BINARY);
            non_zero_right = count_number_of_no_zero_pixels(bin_left);
            cout << "right number " << non_zero_right << endl;

            dst_left = mat_to_eigen(left_temp_otsu);
            dst_right = mat_to_eigen(right_temp_otsu);

            int dilation_elem = 0;
            int dilation_size = 0;

            int dilation_type = 0;

            if( dilation_elem == 0 ){ dilation_type = MORPH_RECT; }
            else if( dilation_elem == 1 ){ dilation_type = MORPH_CROSS; }
            else if( dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }

            Mat element = getStructuringElement( dilation_type,
                                 Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                 Point( dilation_size, dilation_size ) );

            Mat left_temp;
            Mat right_temp;

            dilate( left_temp_otsu, left_temp, element );
            dilate( right_temp_otsu, right_temp, element );

            dst_left_dilated = mat_to_eigen(left_temp_otsu);
            dst_right_dilated = mat_to_eigen(right_temp_otsu);


        }


        save_ori_images(left_view, right_view, index_img, "..//Oricy//", stm.experience_name, stm.sub_experience_name, false, false);

        if(TYPE_OF_FEATURES::PHASE_CONGRUENCY == stm.tof)
        {
            matching_ir_stereo_mat_pair( left_view , right_view, pcr, matchings_stereo, stm, index_img, stm.sub_experience_name, dst_left_dilated, dst_right_dilated);
            save_pc_images(pcr, index_img, "..//ResultsPC//", stm.experience_name, stm.sub_experience_name);
        }
        else
        {
            cv_stereo_match(left_view, right_view, matchings_stereo, stm.tof, stm.verbose);
        }



        string dir_matches = "..//" + get_string_from_tof_enum(stm.tof) + "_matches";
        create_directory(dir_matches);


        MatrixXf left_eigen = mat_to_eigen(left_view);

        MatrixXf right_eigen = mat_to_eigen(right_view);

        int ncols = left_eigen.cols();
        int nrows = left_eigen.rows();


        disparity_3d_projection(matchings_stereo, list_3d_points);

        std::vector<stereo_match> list_projection;

        list_reprojection_save<float>(list_projection, matchings_stereo, list_3d_points);

        if(stm.verbose)
        {
            cout << "taille list_projection " << list_projection.size()  <<  endl;
        }

        std::vector<markers> list_markers;

        segment_objects(nrows, ncols, dst_left, list_markers, stm.verbose);

        std::vector<markers> list_correct_markers;

        projected_objects_image list_objects;



        int nb_pixels_here = 0;
        min_disp = get_objects(list_markers, list_projection, stm.surface_min_person, list_objects, list_correct_markers, nb_pixels_here, min_disp);

        list_markers = list_correct_markers;

        std::vector<float> mean_z(list_objects.size(), 0);

        projected_objects_image list_objects_wo_outliers;

        float nb_matches_here = 0;

        if(stm.del_outliers)
        {
            if( retour_occlusion == 0)
            delete_outliers_from(stm.surface_min_person, list_objects, list_objects_wo_outliers, mean_z, stm.verbose);
            else
                list_objects_wo_outliers = list_objects;
        }
        else
        {
            list_objects_wo_outliers = list_objects;
        }



        int nb_matches = generate_view_from_list_of_two_points<float>(list_objects_wo_outliers, nrows, ncols, path_at_this_index
                                                     ,  path_at_this_index_left, path_at_this_index_right, index_img, stm.verbose);


        auto start_saving = std::chrono::high_resolution_clock::now();
        cout << endl << "saving start " << endl;
        if(stm.save_images)
        {
            //cout << "save" << endl;
            save_stereo_images_markers(list_markers, list_projection,  nrows, ncols, index_img, stm.verbose, "..//Matching_on", stm.experience_name, stm.sub_experience_name);
            save_stereo_segmentation_results(dst_left, dst_right,  stm.eps,matchings_stereo, index_img, "..//Segmentation_results//", stm.experience_name, stm.sub_experience_name);
            save_pc_images(pcr, index_img, "..//ResultsPC//", stm.experience_name, stm.sub_experience_name);
            std::string path_to_save = ".";
            path_to_save = path_to_save + "//nb_matches_" + to_string(stm.eps) + "_" + stm.sub_experience_name + "_.csv";
            ofstream log_detect (path_to_save,  std::ios_base::app);

            log_detect << index_img << ";" << nb_matches << endl ;

            save_image_from_z(list_objects_wo_outliers, left_eigen, nrows, ncols,index_img ,
                              get_string_from_tof_enum(stm.tof), 0.1, "..//Results", stm.experience_name, stm.sub_experience_name, stm.eps);

        }
        auto end_saving = std::chrono::high_resolution_clock::now();
        cout << "saving end " << duration_cast<milliseconds>(end_saving - start_saving).count() << endl << endl;


        matchings_stereo.clear();

        if(stm.verbose)
        {
            cout << "--------------------------------------------------------------------------------------------------------------------------------------" << endl;
            cout << endl << endl << endl;
        }

        //return;


    }
}


}
