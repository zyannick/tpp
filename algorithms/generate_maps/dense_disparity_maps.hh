#pragma once

#include "sparse_disparity_maps.hh"
#include "algorithms/segmentation.hh"
#include "core.hpp"
#include "algorithms/fitting.hh"
#include "global_utils.hh"


namespace tpp
{

void dense_disparity_map()
{
    Mat mat = imread("pom.png", 0);
    MatrixXf img = mat_to_eigen(mat);
    MatrixXf img_res = efficient_euclidian_distance_transform(img);
    normalize_matrix(img_res,  255);
    Mat res = eigen_to_mat(img_res);
    imwrite("resultat.png", res);
}


void dense_disparity_map_pairs()
{
    Mat mat = imread("pom.png", 0);
    MatrixXf img = mat_to_eigen(mat);
    MatrixXf img_res = efficient_euclidian_distance_transform(img);
    normalize_matrix(img_res,  255);
    Mat res = eigen_to_mat(img_res);
    imwrite("resultat.png", res);
}



void center_grave_hot_points(size_t nimages, std::vector<string> imageListCam1, std::vector<string> imageListCam2, std::vector<stereo_match>& matchings_stereo, int scale_factor = 4, int wind_gravity_center = 6,
                             bool with_map_ref = false, Mat left_view_ref = Mat() , Mat right_view_ref = Mat()  )
{
    for(size_t index_img = 0 ;index_img < nimages ; index_img++)
    {
        Mat left_view_16;
        Mat left_view;

        Mat right_view_16;
        Mat right_view;


        left_view_16 = imread(imageListCam1[index_img], CV_16U);

        double minVal, maxVal;
        Point minLoc, maxLoc;
        minMaxLoc(left_view_16, &minVal, &maxVal, &minLoc, &maxLoc);
        left_view_16.convertTo(left_view, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));

        if( scale_factor > 1)
            cv::resize(left_view, left_view, cv::Size(), scale_factor, scale_factor , INTER_CUBIC );

        const int nrows = left_view.rows;
        const int ncols = left_view.cols;

        Mat diff_left;
        if(with_map_ref)
        {
            absdiff(left_view, left_view_ref, diff_left);
            left_view = diff_left;
        }


        minMaxLoc(left_view, &minVal, &maxVal, &minLoc, &maxLoc);
        Point2f coord_gc_left, coord_left;
        coord_left = maxLoc;
        get_local_maxima_gc(wind_gravity_center, left_view, maxLoc, coord_gc_left);




        right_view_16 = imread(imageListCam2[index_img], CV_16U);
        minMaxLoc(right_view_16, &minVal, &maxVal, &minLoc, &maxLoc);
        right_view_16.convertTo(right_view, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));

        if( scale_factor > 1)
            cv::resize(right_view, right_view, cv::Size(), scale_factor, scale_factor , INTER_CUBIC );

        Mat diff_right;
        if(with_map_ref)
        {
            absdiff(right_view, right_view_ref, diff_right);
            right_view = diff_right;
        }


        minMaxLoc(right_view, &minVal, &maxVal, &minLoc, &maxLoc);
        Point2f coord_gc_right, coord_right;
        coord_right = maxLoc;
        get_local_maxima_gc(wind_gravity_center, right_view, maxLoc, coord_gc_right);

        if( fabs(coord_right.y - coord_left.y) < 5 )
        {
            Vector2d left_pt;
            Vector2d right_pt;

            left_pt = Vector2d(coord_gc_left.y, coord_gc_left.x);
            right_pt = Vector2d(coord_gc_right.y, coord_gc_right.x);

            stereo_match st(left_pt, right_pt, 0);
            matchings_stereo.push_back(st);

            //cout << "Points matching before : " << coord_left  << "  " << coord_right <<   "      Points matching after : " << coord_gc_left  << "  " << coord_gc_right  << endl;

        }


    }
}



void dense_segmentation_ground_and_n_walls(string experience_name, size_t nb_walls = 2, int scale_factor = 4, int wind_gravity_center = 6,
                                           bool marked = true, bool with_map_ref = false)
{

    marked = false;

    std::vector<string> imageListCam1_ref, imageListCam2_ref;
    string inputFilename;

    if(with_map_ref)
    {
        imageListCam1_ref = get_list_of_files("..//Images//Ref//Camera1");
        imageListCam2_ref = get_list_of_files("..//Images//Ref//Camera2");
    }

    size_t nimages_ref = imageListCam1_ref.size();

    Mat left_view_16_ref;
    Mat left_view_ref;

    Mat right_view_16_ref;
    Mat right_view_ref;

    if(with_map_ref)
        for(size_t index_img = 0 ;index_img < nimages_ref ; index_img++)
        {


            left_view_16_ref = imread(imageListCam1_ref[index_img], CV_16U);
            double minVal, maxVal;
            Point minLoc, maxLoc;
            minMaxLoc(left_view_16_ref, &minVal, &maxVal, &minLoc, &maxLoc);
            left_view_16_ref.convertTo(left_view_ref, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));

            right_view_16_ref = imread(imageListCam2_ref[index_img], CV_16U);
            minMaxLoc(right_view_16_ref, &minVal, &maxVal, &minLoc, &maxLoc);
            right_view_16_ref.convertTo(right_view_ref, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));

        }

    std::vector<string> imageListCam1_ground, imageListCam2_ground;

    std::vector<std::vector<string>> list_path_to_walls_cam1(nb_walls), list_path_to_walls_cam2(nb_walls);

    std::vector<std::vector<string>> list_paths_to_images_cam1(nb_walls), list_paths_to_images_cam2(nb_walls);

    string path_to_ground_camera1 = string("..//ground_segmentation_dataset//").append(experience_name).append("//").append(experience_name).append("Ground").append(string("//Camera1"));
    string path_to_ground_camera2 = string("..//ground_segmentation_dataset//").append(experience_name).append("//").append(experience_name).append("Ground").append(string("//Camera2"));


    imageListCam1_ground = get_list_of_files(path_to_ground_camera1);
    imageListCam2_ground = get_list_of_files(path_to_ground_camera2);

    size_t nb_images_all_walls = 0;


    for(size_t index_wall = 1; index_wall <= nb_walls; index_wall++)
    {
        namespace fs = std::experimental::filesystem;
        string path_to_wall_camera1 = string("..//ground_segmentation_dataset//").append(experience_name).append("//").append(experience_name).append("Wall").append(to_string(index_wall)).append(string("//Camera1"));
        string path_to_wall_camera2 = string("..//ground_segmentation_dataset//").append(experience_name).append("//").append(experience_name).append("Wall").append(to_string(index_wall)).append(string("//Camera2"));

        assert(fs::is_directory(path_to_wall_camera1) && fs::exists(path_to_wall_camera1) && fs::is_directory(path_to_wall_camera2) && fs::exists(path_to_wall_camera2));

        list_paths_to_images_cam1[index_wall - 1] = get_list_of_files(path_to_wall_camera1);
        list_paths_to_images_cam2[index_wall - 1] = get_list_of_files(path_to_wall_camera2);

        nb_images_all_walls += list_paths_to_images_cam1[index_wall - 1].size();
    }




    stereo_params_cv stereo_par;
    stereo_par.retreive_values();

    size_t nimages_ground = imageListCam1_ground.size();


    cout << "number of pairs for ground detection  " << nimages_ground << endl ;
    cout << "number of walls " << nb_walls << "  number of images of all walls " << nb_images_all_walls <<  endl ;

    std::vector<stereo_match> matchings_stereo_ground;
    std::vector<Vector3d> list_of_3d_points_ground;

    std::vector<std::vector<stereo_match>> matchings_stereo_wall(nb_walls);
    std::vector<std::vector<Vector3d>> list_of_3d_points_wall(nb_walls);

    string dir_name("./3d_points_3_planes/");
    create_directory(dir_name);


    std::string filename_3d_points_ground = dir_name;
    filename_3d_points_ground += (experience_name + "ground_3d_points.txt");
    ofstream save_3d_points_ground (filename_3d_points_ground);



    std::vector<Vector3d> list_3d_points_ground;
    std::vector<std::vector<Vector3d>> list_3d_points_wall(nb_walls);



    //cout << "get points of ground" << endl << endl;
    center_grave_hot_points(nimages_ground,imageListCam1_ground, imageListCam2_ground, matchings_stereo_ground, scale_factor, wind_gravity_center);
    disparity_3d_projection(matchings_stereo_ground, list_3d_points_ground);

    //cout << "get points of walls" << endl;
    for(size_t index_wall = 0; index_wall < nb_walls ; index_wall++)
    {
        //cout << "wall number " << index_wall << endl;

        center_grave_hot_points(list_paths_to_images_cam1[index_wall].size(),list_paths_to_images_cam1[index_wall],
                                list_paths_to_images_cam2[index_wall], matchings_stereo_wall[index_wall], scale_factor, wind_gravity_center);
        disparity_3d_projection(matchings_stereo_wall[index_wall], list_3d_points_wall[index_wall]);
    }

    //cout << endl << endl;


    float val_min = 100000000;
    float val_max = 0;

    //cout << "taille ground " << matchings_stereo_ground.size() << endl;

    //cout << "taille wall " << matchings_stereo_wall.size() << endl;

    size_t index_matche = 0;
    for(stereo_match mt:  matchings_stereo_ground)
    {
        int x = int(mt.first_point(1));
        int y = int(mt.first_point(0));
        float val_z = float(list_3d_points_ground[index_matche](2));

                if(double(val_z) > 0 && double(val_z) < 10000.0)
        {
                //cout << "val_z " << val_z << endl;

                list_of_3d_points_ground.push_back(list_3d_points_ground[index_matche]);

                save_3d_points_ground << std::fixed << std::setprecision(5) << list_3d_points_ground[index_matche](0) << ";" << list_3d_points_ground[index_matche](1)  << ";"  << list_3d_points_ground[index_matche](2) << "\n";

                //cout <<  list_3d_points_ground[index_matche](0) << ";" << list_3d_points_ground[index_matche](1)  << ";"  << list_3d_points_ground[index_matche](2) << "\n";

    }
                index_matche++;
    }
    save_3d_points_ground.close();



    for(size_t index_wall = 0; index_wall < nb_walls ; index_wall++)
    {
        index_matche = 0;
        std::string filename_3d_points_wall = dir_name;
        filename_3d_points_wall += experience_name +  "wall_3d_points_" + to_string(index_wall + 1) + ".txt";
        ofstream save_3d_points_wall (filename_3d_points_wall);
        for(stereo_match mt:  matchings_stereo_wall[index_wall])
        {
            int x = int(mt.first_point(1));
            int y = int(mt.first_point(0));
            float val_z = float(list_3d_points_wall[index_wall][index_matche](2));
                    if(double(val_z) > 0 && double(val_z) < 10000.0)
            {
                    //cout << "val_z " << val_z << endl;

                    list_of_3d_points_wall[index_wall].push_back(list_3d_points_wall[index_wall][index_matche]);


                    save_3d_points_wall << std::fixed << std::setprecision(5) << list_3d_points_wall[index_wall][index_matche](0)
                                           << ";" << list_3d_points_wall[index_wall][index_matche](1)  << ";"  << list_3d_points_wall[index_wall][index_matche](2) << "\n";

                    //cout <<  list_3d_points_wall[index_wall][index_matche](0) << ";" << list_3d_points_wall[index_wall][index_matche](1)  << ";"  << list_3d_points_wall[index_wall][index_matche](2) << "\n";

        }
                    index_matche++;
        }
        save_3d_points_wall.close();
    }


    std::vector<Vector3d> wall_plane(nb_walls);

    std::vector<Vector4d> abcd(nb_walls);

    std::vector<VectorXf> list_error_per_plan_per_point;

    for(size_t index_wall = 0 ; index_wall < nb_walls ; index_wall++)
    {
        wall_plane[index_wall] = best_plane_from_points_svd(list_of_3d_points_wall[index_wall]);
        abcd[index_wall] = Vector4d(wall_plane[index_wall].y(), wall_plane[index_wall].x(), -1, wall_plane[index_wall].z());
    }


    Vector4d ransac_ground_plane = ransac_plane_fitting_ground_n_walls(list_of_3d_points_ground, abcd);


    std::string save_regression_ground = dir_name;
    save_regression_ground += (experience_name + "_3planes_ground.txt");
    ofstream myfile_regression_ground (save_regression_ground);



    if (myfile_regression_ground.is_open() )
    {
        cout << "best plane " << ransac_ground_plane(0) << ";" << ransac_ground_plane(1) << ";" << ransac_ground_plane(2) << ";" << ransac_ground_plane(3)  << "\n";
        //cout << "plane " << wall_plane(0) << ";" << wall_plane(1) << ";" << wall_plane(2)  << "\n";
        myfile_regression_ground << std::fixed << std::setprecision(8) << ransac_ground_plane(0) << ";" << ransac_ground_plane(1) << ";" << ransac_ground_plane(2) << ";" << ransac_ground_plane(3)  << "\n";

        for(size_t index_wall = 0 ; index_wall < nb_walls ; index_wall++)
        {
            std::string save_regression_wall = dir_name;
            save_regression_wall += experience_name + "wall_" + to_string(index_wall + 1) + (".txt");
            ofstream myfile_regression_wall (save_regression_wall);

            if(myfile_regression_wall.is_open())
            {
                myfile_regression_wall << std::fixed << std::setprecision(8) << abcd[index_wall](0) << ";" << abcd[index_wall](1) << ";" << abcd[index_wall](2) << ";" << abcd[index_wall](3)  << "\n";
            }
            myfile_regression_wall.close();
        }

    }

    myfile_regression_ground.close();

    cout << "-------------------------------------------------------------------------" << endl << endl << endl;



}


void dense_segmentation_ground_and_wall(string experience_name,int scale_factor = 4, int wind_gravity_center = 6,bool save = true,
                                        bool rect = true,MATCHING_SIMILARITY_METHOD dist_method = MATCHING_SIMILARITY_METHOD::LADES_SIMILARITY,
                                        SUB_PIXEL_MATCHING sub_pixel = SUB_PIXEL_MATCHING::PHASE_CORRELATION, bool resized = true, bool marked = true,
                                        bool with_map_ref = false)
{

    marked = false;

    std::vector<string> imageListCam1_ref, imageListCam2_ref;
    string inputFilename;

    if(with_map_ref)
    {
        imageListCam1_ref = get_list_of_files("..//Images//Ref//Camera1");
        imageListCam2_ref = get_list_of_files("..//Images//Ref//Camera2");
    }

    size_t nimages_ref = imageListCam1_ref.size();

    Mat left_view_16_ref;
    Mat left_view_ref;

    Mat right_view_16_ref;
    Mat right_view_ref;

    if(with_map_ref)
        for(size_t index_img = 0 ;index_img < nimages_ref ; index_img++)
        {


            left_view_16_ref = imread(imageListCam1_ref[index_img], CV_16U);
            double minVal, maxVal;
            Point minLoc, maxLoc;
            minMaxLoc(left_view_16_ref, &minVal, &maxVal, &minLoc, &maxLoc);
            left_view_16_ref.convertTo(left_view_ref, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));

            right_view_16_ref = imread(imageListCam2_ref[index_img], CV_16U);
            minMaxLoc(right_view_16_ref, &minVal, &maxVal, &minLoc, &maxLoc);
            right_view_16_ref.convertTo(right_view_ref, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));

        }

    std::vector<string> imageListCam1_ground, imageListCam2_ground;

    std::vector<string> imageListCam1_wall, imageListCam2_wall;

    string path_to_ground_camera1 = string("..//ground_segmentation_dataset//").append(experience_name).append("//").append(experience_name).append("Ground").append(string("//Camera1"));
    string path_to_ground_camera2 = string("..//ground_segmentation_dataset//").append(experience_name).append("//").append(experience_name).append("Ground").append(string("//Camera2"));


    imageListCam1_ground = get_list_of_files(path_to_ground_camera1);
    imageListCam2_ground = get_list_of_files(path_to_ground_camera2);

    string path_to_wall_camera1 = string("..//ground_segmentation_dataset//").append(experience_name).append("//").append(experience_name).append("Wall1").append(string("//Camera1"));
    string path_to_wall_camera2 = string("..//ground_segmentation_dataset//").append(experience_name).append("//").append(experience_name).append("Wall1").append(string("//Camera2"));


    imageListCam1_wall = get_list_of_files(path_to_wall_camera1);
    imageListCam2_wall = get_list_of_files(path_to_wall_camera2);

    stereo_params_cv stereo_par;
    stereo_par.retreive_values();

    size_t nimages_ground = imageListCam1_ground.size();
    size_t nimages_wall = imageListCam1_wall.size();

    cout << "number of pairs for ground detection  " << nimages_ground << endl ;
    cout << "number of pairs for wall detection  " << nimages_wall << endl ;

    std::vector<stereo_match> matchings_stereo_ground;
    std::vector<stereo_match> list_projections_ground;
    std::vector<Vector3d> list_of_3d_points_ground;

    std::vector<stereo_match> matchings_stereo_wall;
    std::vector<stereo_match> list_projections_wall;
    std::vector<Vector3d> list_of_3d_points_wall;

    string dir_name("./3d_points_2_planes/");
    create_directory(dir_name);


    std::string filename_3d_points_ground = dir_name;
    filename_3d_points_ground +=  (experience_name + "_ground_3d_points.txt");
    ofstream save_3d_points_ground (filename_3d_points_ground);

    std::string filename_3d_points_wall = dir_name;
    filename_3d_points_wall += (experience_name + "_face_wall_3d_points.txt");
    ofstream save_3d_points_wall (filename_3d_points_wall);

    std::vector<Vector3d> list_3d_points_ground;
    std::vector<Vector3d> list_3d_points_wall;



    center_grave_hot_points(nimages_ground,imageListCam1_ground, imageListCam2_ground, matchings_stereo_ground,scale_factor , wind_gravity_center);
    center_grave_hot_points(nimages_wall,imageListCam1_wall, imageListCam2_wall, matchings_stereo_wall,scale_factor , wind_gravity_center);

    disparity_3d_projection(matchings_stereo_ground, list_3d_points_ground);
    disparity_3d_projection(matchings_stereo_wall, list_3d_points_wall);

    float val_min = 100000000;
    float val_max = 0;

    //cout << "taille ground " << matchings_stereo_ground.size() << endl;

    //cout << "taille wall " << matchings_stereo_wall.size() << endl;

    size_t index_matche = 0;
    for(stereo_match mt:  matchings_stereo_ground)
    {
        int x = int(mt.first_point(1));
        int y = int(mt.first_point(0));
        float val_z = float(list_3d_points_ground[index_matche](2));

                if(double(val_z) > 0 && double(val_z) < 10000.0)
        {
                //cout << "val_z " << val_z << endl;

                list_of_3d_points_ground.push_back(list_3d_points_ground[index_matche]);

                save_3d_points_ground << std::fixed << std::setprecision(5) << list_3d_points_ground[index_matche](0) << ";" << list_3d_points_ground[index_matche](1)  << ";"  << list_3d_points_ground[index_matche](2) << "\n";

                //cout <<  list_3d_points_ground[index_matche](0) << ";" << list_3d_points_ground[index_matche](1)  << ";"  << list_3d_points_ground[index_matche](2) << "\n";

    }
                index_matche++;
    }


    index_matche = 0;
    for(stereo_match mt:  matchings_stereo_wall)
    {
        int x = int(mt.first_point(1));
        int y = int(mt.first_point(0));
        float val_z = float(list_3d_points_wall[index_matche](2));
                if(double(val_z) > 0 && double(val_z) < 10000.0)
        {
                //cout << "val_z " << val_z << endl;

                list_of_3d_points_wall.push_back(list_3d_points_wall[index_matche]);

                save_3d_points_wall << std::fixed << std::setprecision(5) << list_3d_points_wall[index_matche](0) << ";" << list_3d_points_wall[index_matche](1)  << ";"  << list_3d_points_wall[index_matche](2) << "\n";

                //cout <<  list_3d_points_wall[index_matche](0) << ";" << list_3d_points_wall[index_matche](1)  << ";"  << list_3d_points_wall[index_matche](2) << "\n";

    }
                index_matche++;
    }


    save_3d_points_ground.close();
    save_3d_points_wall.close();

    //ransac_plane_fitting(list_of_3d_points_wall);

    //ransac_plane_fitting(list_of_3d_points_wall);


    Vector3d wall_plane = best_plane_from_points_svd(list_of_3d_points_wall);

    Vector4d abcd(wall_plane.y(), wall_plane.x(), -1, wall_plane.z());

    //Vector4d ransac_ground_plane = ransac_plane_fitting_ground_wall(list_of_3d_points_ground, abcd);

    Vector3d ground_plane = best_plane_from_points_svd(list_of_3d_points_ground);

    Vector4d ransac_ground_plane(ground_plane.y(), ground_plane.x(), -1, ground_plane.z());


    std::string save_regression_ground = dir_name;
    save_regression_ground += (experience_name + "_equation_ground.txt");
    ofstream myfile_regression_ground (save_regression_ground);

    std::string save_regression_wall = dir_name;
    save_regression_wall += (experience_name + "_equation_face_wall.txt");
    ofstream myfile_regression_wall (save_regression_wall);


    if (myfile_regression_ground.is_open() && myfile_regression_wall.is_open())
    {
        cout << "best plane " << ransac_ground_plane(0) << ";" << ransac_ground_plane(1) << ";" << ransac_ground_plane(2) << ";" << ransac_ground_plane(3)  << "\n";
        //cout << "plane " << wall_plane(0) << ";" << wall_plane(1) << ";" << wall_plane(2)  << "\n";
        myfile_regression_ground << std::fixed << std::setprecision(8) << ransac_ground_plane(0) << ";" << ransac_ground_plane(1) << ";" << ransac_ground_plane(2) << ";" << ransac_ground_plane(3)  << "\n";

        myfile_regression_wall << std::fixed << std::setprecision(8) << abcd(0) << ";" << abcd(1) << ";" << abcd(2) << ";" << abcd(3)  << "\n";
    }

    myfile_regression_ground.close();

    myfile_regression_wall.close();

    cout << "-------------------------------------------------------------------------" << endl << endl << endl;



}



void dense_segmentation_ground(string experience_name, int scale_factor = 4, int wind_gravity_center = 6,bool save = true,
                               bool rect = true,MATCHING_SIMILARITY_METHOD dist_method = MATCHING_SIMILARITY_METHOD::LADES_SIMILARITY,
                               SUB_PIXEL_MATCHING sub_pixel = SUB_PIXEL_MATCHING::PHASE_CORRELATION, bool resized = true, bool marked = true,
                               bool with_map_ref = false)
{

    marked = false;

    std::vector<string> imageListCam1_ref, imageListCam2_ref;
    string inputFilename;

    if(with_map_ref)
    {
        imageListCam1_ref = get_list_of_files("..//Images//Ref//Camera1");
        imageListCam2_ref = get_list_of_files("..//Images//Ref//Camera2");
    }

    size_t nimages_ref = imageListCam1_ref.size();

    Mat left_view_16_ref;
    Mat left_view_ref;

    Mat right_view_16_ref;
    Mat right_view_ref;

    if(with_map_ref)
        for(size_t index_img = 0 ;index_img < nimages_ref ; index_img++)
        {


            left_view_16_ref = imread(imageListCam1_ref[index_img], CV_16U);
            double minVal, maxVal;
            Point minLoc, maxLoc;
            minMaxLoc(left_view_16_ref, &minVal, &maxVal, &minLoc, &maxLoc);
            left_view_16_ref.convertTo(left_view_ref, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));

            right_view_16_ref = imread(imageListCam2_ref[index_img], CV_16U);
            minMaxLoc(right_view_16_ref, &minVal, &maxVal, &minLoc, &maxLoc);
            right_view_16_ref.convertTo(right_view_ref, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));

        }

    std::vector<string> imageListCam1, imageListCam2;

    if(!marked)
    {
        string path_to_ground_camera1 = string("..//ground_segmentation_dataset//").append(experience_name).append("//").append(experience_name).append("Ground").append(string("//Camera1"));
        string path_to_ground_camera2 = string("..//ground_segmentation_dataset//").append(experience_name).append("//").append(experience_name).append("Ground").append(string("//Camera2"));


        imageListCam1 = get_list_of_files(path_to_ground_camera1);
        imageListCam2 = get_list_of_files(path_to_ground_camera2);



    }
    else {
        imageListCam1 = get_list_of_files("..//Images//GetGroundMarked//Camera1");
        imageListCam2 = get_list_of_files("..//Images//GetGroundMarked//Camera2");
    }




    stereo_params_cv stereo_par;
    stereo_par.retreive_values();

    float eps = float(0.01);

    size_t nimages = imageListCam1.size();

    cout << "number of pairs for ground detection  " << nimages << endl ;

    std::vector<stereo_match> matchings_stereo;
    phase_congruency_result<MatrixXf> pcr;

    std::vector<stereo_match> list_projections;

    std::vector<Vector3d> list_of_3d_points;

    string dir_name("./3d_points/");

    create_directory(dir_name);

    std::string save_3d_plane_points = dir_name;
    save_3d_plane_points += (experience_name + "only_ground_plane_80x60_3d_points.txt");
    ofstream myfile_plane (save_3d_plane_points);

    std::string save_3d_points = dir_name;
    save_3d_points +=  (experience_name + "only_ground_plane_3d_points.txt");
    ofstream save_3d_points_file (save_3d_points);

    std::vector<Vector3d> list_3d_points;


    for(size_t index_img = 0 ;index_img < nimages ; index_img++)
    {
        Mat left_view_16;
        Mat left_view;

        Mat right_view_16;
        Mat right_view;

        if(!marked)
        {
            left_view_16 = imread(imageListCam1[index_img], CV_16U);

            double minVal, maxVal;
            Point minLoc, maxLoc;
            minMaxLoc(left_view_16, &minVal, &maxVal, &minLoc, &maxLoc);
            left_view_16.convertTo(left_view, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));

            if(scale_factor > 1)
                cv::resize(left_view, left_view, cv::Size(), scale_factor, scale_factor , INTER_CUBIC );

            const int nrows = left_view.rows;
            const int ncols = left_view.cols;

            //GaussianBlur( left_view, left_view, Size( 5, 5 ), 0, 0 );
            Mat diff_left;
            if(with_map_ref)
            {
                absdiff(left_view, left_view_ref, diff_left);
                left_view = diff_left;
            }

            //Mat img_interpolate = Mat::zeros(nrows, ncols, CV_64FC1);
            //bilinear_interpolation(left_view, img_interpolate);
            //left_view = img_interpolate;
            minMaxLoc(left_view, &minVal, &maxVal, &minLoc, &maxLoc);
            Point2f coord_gc_left, coord_left;
            coord_left = maxLoc;
            get_local_maxima_gc(wind_gravity_center, left_view, maxLoc, coord_gc_left);

            right_view_16 = imread(imageListCam2[index_img], CV_16U);
            minMaxLoc(right_view_16, &minVal, &maxVal, &minLoc, &maxLoc);
            right_view_16.convertTo(right_view, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));

            if(scale_factor > 1)
                cv::resize(right_view, right_view, cv::Size(), scale_factor, scale_factor , INTER_CUBIC );

            Mat diff_right;
            if(with_map_ref)
            {
                absdiff(right_view, right_view_ref, diff_right);
                right_view = diff_right;
            }

            //GaussianBlur( right_view, right_view, Size( 5, 5 ), 0, 0 );
            //img_interpolate = Mat::zeros(nrows, ncols, CV_64FC1);
            //bilinear_interpolation(right_view, img_interpolate);
            //right_view = img_interpolate;
            minMaxLoc(right_view, &minVal, &maxVal, &minLoc, &maxLoc);
            Point2f coord_gc_right, coord_right;
            coord_right = maxLoc;
            get_local_maxima_gc(wind_gravity_center, right_view, maxLoc, coord_gc_right);

            if( fabs(coord_right.y - coord_left.y) < 5 )
            {
                Vector2d left_pt;
                Vector2d right_pt;

                left_pt = Vector2d(coord_gc_left.y, coord_gc_left.x);
                right_pt = Vector2d(coord_gc_right.y, coord_gc_right.x);

                stereo_match st(left_pt, right_pt, 0);
                matchings_stereo.push_back(st);

                //cout << "Points matching before : " << coord_left  << "  " << coord_right <<   "      Points matching after : " << coord_gc_left  << "  " << coord_gc_right  << endl;

            }
            //disparity_map_pairs(pcr, matchings_stereo, left_view , right_view, stereo_par, wind_row, wind_col, eps, rect, int(index_img) ,0, dist_method,sub_pixel );
        }
        else
        {
            left_view = imread(imageListCam1[index_img], 1);
            right_view = imread(imageListCam2[index_img], 1);

            const int nrows = left_view.rows;
            const int ncols = left_view.cols;

            Vector2d left_pt;
            Vector2d right_pt;

            for(int row = 0 ; row < nrows; row ++)
            {
                for(int col = 0; col < ncols ; col++)
                {
                    Vec3b left_value = left_view.at<Vec3b>(row,col);
                    Vec3b right_value = right_view.at<Vec3b>(row,col);

                    if(left_value[2] == 237 && left_value[1] == 28 && left_value[0]==36)
                    {
                        left_pt.x() = col;
                        left_pt.y() = row;
                    }

                    if(right_value[2] == 237 && right_value[1] == 28 && right_value[0]==36)
                    {
                        right_pt.x() = col;
                        right_pt.y() = row;
                    }
                }
            }
            stereo_match st(left_pt, right_pt, 0);
            matchings_stereo.push_back(st);
        }

    }




    disparity_3d_projection(matchings_stereo, list_3d_points);



    size_t index_matche = 0;



    float val_min = 100000000;
    float val_max = 0;


    //cout << "taille " << matchings_stereo.size() << endl;

    for(stereo_match mt:  matchings_stereo)
    {
        int x = int(mt.first_point(1));
        int y = int(mt.first_point(0));
        float val_z = float(list_3d_points[index_matche](2));

                if(
                    double(val_z) > 0 && double(val_z) < 10000.0
                    )
        {
                //cout << "val_z " << val_z << endl;

                list_of_3d_points.push_back(list_3d_points[index_matche]);

                myfile_plane << std::fixed << std::setprecision(5) << y << ";" << x  << ";"  << list_3d_points[index_matche](2) << "\n";

                save_3d_points_file << std::fixed << std::setprecision(5) << list_3d_points[index_matche](0) << ";" << list_3d_points[index_matche](1)  << ";"  << list_3d_points[index_matche](2) << "\n";

                //cout <<  list_3d_points[index_matche](0) << ";" << list_3d_points[index_matche](1)  << ";"  << list_3d_points[index_matche](2) << "\n";

    }
                index_matche++;
    }


    myfile_plane.close();
    save_3d_points_file.close();

    Vector3d ground_plane = best_plane_from_points_svd(list_of_3d_points);

    Vector4d best_plane = ransac_plane_fitting(list_of_3d_points);



    std::string save_regression_plane = dir_name;
    save_regression_plane += (experience_name + "_only_ground_plane.txt");
    ofstream myfile (save_regression_plane);

    if (myfile.is_open())
    {
        cout << "best plane " << best_plane(0) << ";" << best_plane(1) << ";" << best_plane(2) << ";" << best_plane(3)  << "\n";
        cout << "plane " << ground_plane(0) << ";" << ground_plane(1) << ";" << ground_plane(2)  << "\n";
        myfile << std::fixed << std::setprecision(8) << best_plane(0) << ";" << best_plane(1) << ";" << best_plane(2) << ";" << best_plane(3)  << "\n";
    }

    myfile.close();

    cout << "-------------------------------------------------------------------------" << endl << endl << endl;



}



void ground_plane_detection_with_phase_congruency(int wind_row, int wind_col , string experience_name, int scale_factor = 4,  bool save = true,
                                                  bool rect = true,MATCHING_SIMILARITY_METHOD dist_method = MATCHING_SIMILARITY_METHOD::LADES_SIMILARITY,
                                                  SUB_PIXEL_MATCHING_PRECISE sub_pixel = SUB_PIXEL_MATCHING_PRECISE::PHASE_CORRELATION_FORROSH, bool resized = true, bool plane_z_mean = false)
{
    std::vector<string> imageListCam1, imageListCam2;
    string inputFilename;

    string path_to_ground_camera1 = string("..//").append(experience_name).append("//").append(experience_name).append("Ground").append(string("//Camera1"));
    string path_to_ground_camera2 = string("..//").append(experience_name).append("//").append(experience_name).append("Ground").append(string("//Camera2"));

    imageListCam1 = get_list_of_files(path_to_ground_camera1);
    imageListCam2 = get_list_of_files(path_to_ground_camera2);

    stereo_params_cv stereo_par;
    stereo_par.retreive_values();

    float eps = float(0.01);

    size_t nimages = imageListCam1.size();

    cout << "number of pairs " << nimages << endl ;
    cout << "-------------------------------------------------------------------------" << endl << endl << endl;

    std::vector<stereo_match> matchings_stereo;
    phase_congruency_result<MatrixXf> pcr;

    string dir_name = "./3d_points/";

    create_directory(dir_name);

    std::string save_3d_points = dir_name;
    save_3d_points.append(string("ground_segmented_all")).append(".txt");
    ofstream myfile_3d_points (save_3d_points);

    if (!myfile_3d_points.is_open())
        return;

    for(size_t index_img = 0 ;index_img < nimages ; index_img++)
    {
        std::vector<Vector3d> list_3d_points;

        Mat left_view_16 = imread(imageListCam1[index_img], CV_16U);
        Mat left_view;

        double minVal, maxVal;
        Point minLoc, maxLoc;
        minMaxLoc(left_view_16, &minVal, &maxVal, &minLoc, &maxLoc);
        left_view_16.convertTo(left_view, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));

        if(scale_factor > 1)
            cv::resize(left_view, left_view, cv::Size(), scale_factor, scale_factor , INTER_CUBIC );

        Mat right_view_16 = imread(imageListCam2[index_img], CV_16U);
        Mat right_view;

        minMaxLoc(right_view_16, &minVal, &maxVal, &minLoc, &maxLoc);
        right_view_16.convertTo(right_view, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));

        if(scale_factor > 1)
            cv::resize(right_view, right_view, cv::Size(), scale_factor, scale_factor , INTER_CUBIC );

        //disparity_map_pairs(pcr, matchings_stereo, left_view , right_view, stereo_par, wind_row, wind_col, eps, rect, int(index_img) ,0, dist_method,sub_pixel );

        disparity_3d_projection(matchings_stereo, list_3d_points);

        const int nrows = left_view.rows;
        const int ncols = left_view.cols;

        size_t index_matche = 0;

        std::vector<Vector3d> list_points;

        std::vector<Vector3d> list_points_for_plan;

        float val_min = 100000000;
        float val_max = 0;

        for(stereo_match mt:  matchings_stereo)
        {
            int x = int(mt.first_point(1));
            int y = int(mt.first_point(0));
            float val_z = float(list_3d_points[index_matche](2));
                    if(double(val_z) > 0 && double(val_z) < 10000.0)
            {


                    list_points.push_back(Vector3d(val_z,y,x));
                    list_points_for_plan.push_back(Vector3d(float(list_3d_points[index_matche](2)),float(list_3d_points[index_matche](1)),float(list_3d_points[index_matche](0))));
        }
                    index_matche++;
        }


        MatrixXf dst;
        MatrixXf edges_left = pcr.pc_first;


        MatrixXf left_eigen = mat_to_eigen(left_view);
        int th = otsu_segmentation(left_eigen);
        cout << "seuil " << th << endl;
        binarize_image(left_eigen,  dst , th , 0 , 255);


        typedef std::vector<Vector2i> markers;
        std::vector<markers> list_markers;

        Vector3i color_value;

        size_t index = 0;

        for(int row = 0 ; row < nrows; row++)
        {
            for(int col = 0; col < ncols; col++)
            {
                if(int(dst(row,col))==255)
                {
                    markers mark;
                    // take the point in format (y,x)
                    mark.push_back(Vector2i(row, col));
                    populate_marker_segmentation(mark, row, col, dst , index);
                    list_markers.push_back(mark);
                    index++;
                }
            }
        }

        typedef std::vector<Vector3d> _3d_points_segmented;
        std::vector<_3d_points_segmented> list_3d_points_segmented;

        std::vector<markers> list_correct_markers;


        for(int index_marker = 0 ; index_marker < list_markers.size() ; index_marker++)
        {
            markers mark = list_markers[index_marker];
            _3d_points_segmented temp_3d_points;
            for(int k = 0 ; k < mark.size() ; k++)
            {
                Vector2i val_comp = mark[k];
                for(int i = 0; i < list_points.size(); i ++)
                {
                    Vector2i val = (list_points[i]).segment(1,2).cast<int>();
                    float norm = (val_comp - val).cast<float>().norm();
                    if(norm < 1)
                    {
                        temp_3d_points.push_back(list_points_for_plan[i]);
                    }
                }
            }
            if(temp_3d_points.size() > 4)
            {
                list_3d_points_segmented.push_back(temp_3d_points);
                list_correct_markers.push_back(list_markers[index_marker]);
            }

        }

        list_markers = list_correct_markers;



        for(size_t index_marker = 0 ; index_marker < list_markers.size() ; index_marker++)
        {
            for(int idx = 0 ; idx < list_3d_points_segmented[index_marker].size(); idx++)
            {

                myfile_3d_points << std::fixed << std::setprecision(5) << list_3d_points_segmented[index_marker][idx].z() << ";"
                                 << list_3d_points_segmented[index_marker][idx].x() << ";"
                                 << list_3d_points_segmented[index_marker][idx].y() << "\n";

            }
        }

        matchings_stereo.clear();

    }
    myfile_3d_points.close();

}

void ground_segmentation(int wind_row, int wind_col ,string experience_name, size_t nb_walls,bool is_point, float eps, int scale_factor = 4, int wind_gravity_center = 6,  bool save = true,
                         bool rect = true,MATCHING_SIMILARITY_METHOD dist_method = MATCHING_SIMILARITY_METHOD::LADES_SIMILARITY,
                         SUB_PIXEL_MATCHING_PRECISE sub_pixel = SUB_PIXEL_MATCHING_PRECISE::PHASE_CORRELATION_FORROSH, bool resized = true, bool plane_z_mean = false, bool with_map_ref = true)
{


    if(is_point)
    {
        if(nb_walls == 0)
        {
            dense_segmentation_ground(experience_name,scale_factor,wind_gravity_center);
        }
        else if(nb_walls == 1)
        {
            dense_segmentation_ground_and_wall(experience_name,scale_factor,wind_gravity_center);
        }
        else if(nb_walls > 1)
        {
            dense_segmentation_ground_and_n_walls(experience_name , nb_walls,scale_factor,wind_gravity_center);
        }
    }
    else
    {
        ground_plane_detection_with_phase_congruency(wind_row, wind_row ,experience_name,scale_factor);
    }
    cout << endl << endl << endl;
}


void three_rooms_segmentation(int wind_row, int wind_col ,string experience_name, size_t nb_walls,bool is_point, float eps, int scale_factor = 4, int wind_gravity_center = 6,  bool save = true,
                              bool rect = true,MATCHING_SIMILARITY_METHOD dist_method = MATCHING_SIMILARITY_METHOD::LADES_SIMILARITY,
                              SUB_PIXEL_MATCHING_PRECISE sub_pixel = SUB_PIXEL_MATCHING_PRECISE::PHASE_CORRELATION_FORROSH, bool resized = true, bool plane_z_mean = false, bool with_map_ref = true)
{
    if(is_point)
    {
        if(nb_walls == 0)
        {
            dense_segmentation_ground(experience_name,scale_factor,wind_gravity_center);
        }
        else if(nb_walls == 1)
        {
            dense_segmentation_ground_and_wall(experience_name,scale_factor,wind_gravity_center);
        }
        else if(nb_walls > 1)
        {
            dense_segmentation_ground_and_n_walls(experience_name , nb_walls,scale_factor,wind_gravity_center);
        }
    }
    else
    {
        ground_plane_detection_with_phase_congruency(wind_row, wind_row ,experience_name,scale_factor);
    }


    std::vector<string> imageListCam1, imageListCam2;
    string inputFilename;

    std::vector<string> imageListCam1_ref, imageListCam2_ref;


    string path_to_ref_camera1_images = string("..//").append(experience_name).append("//").append(experience_name).append("Ref").append(string("//Camera1"));
    string path_to_ref_camera2_images = string("..//").append(experience_name).append("//").append(experience_name).append("Ref").append(string("//Camera2"));

    double mean_max_val = 0.0;

    if(with_map_ref)
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

    cout << "mean_max_val  " << mean_max_val << endl;



    string path_to_camera1_images = string("..//").append(experience_name).append("//").append(experience_name).append("Person").append(string("//Camera1"));
    string path_to_camera2_images = string("..//").append(experience_name).append("//").append(experience_name).append("Person").append(string("//Camera2"));

    imageListCam1 = get_list_of_files(path_to_camera1_images);
    imageListCam2 = get_list_of_files(path_to_camera2_images);

    stereo_params_cv stereo_par;
    stereo_par.retreive_values();

    size_t nimages = imageListCam1.size();

    cout << "number of pairs " << nimages << endl ;
    cout << "-------------------------------------------------------------------------" << endl << endl << endl;

    std::vector<stereo_match> matchings_stereo;
    phase_congruency_result<MatrixXf> pcr;

    string dir_name = "./3d_points/";

    create_directory(dir_name);

    for(size_t index_img = 0 ;index_img < nimages ; index_img++)
    {
        std::vector<Vector3d> list_3d_points;


        std::string save_3d_points = dir_name;
        save_3d_points.append(to_string(index_img)).append(".txt");
        ofstream myfile_3d_points (save_3d_points);

        if (!myfile_3d_points.is_open())
            continue;

        Mat left_view_16 = imread(imageListCam1[index_img], CV_16U);
        Mat left_view;

        double minVal, maxVal;
        Point minLoc, maxLoc;
        minMaxLoc(left_view_16, &minVal, &maxVal, &minLoc, &maxLoc);
        cout << "min val " << minVal << "    max val " << maxVal << endl;
        left_view_16.convertTo(left_view, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));

        if( scale_factor > 1)
            cv::resize(left_view, left_view, cv::Size(), scale_factor, scale_factor , INTER_CUBIC );

        Mat right_view_16 = imread(imageListCam2[index_img], CV_16U);
        Mat right_view;

        minMaxLoc(right_view_16, &minVal, &maxVal, &minLoc, &maxLoc);
        right_view_16.convertTo(right_view, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));

        if( scale_factor > 1)
            cv::resize(right_view, right_view, cv::Size(), scale_factor, scale_factor , INTER_CUBIC );


        //disparity_map_pairs(pcr, matchings_stereo, left_view , right_view, stereo_par, wind_row, wind_col, eps, rect, int(index_img) ,0, dist_method,sub_pixel );

        disparity_3d_projection(matchings_stereo, list_3d_points);

        const int nrows = left_view.rows;
        const int ncols = left_view.cols;

        cout << "verification nrows = " << nrows << "  ncols = " << ncols << endl;

        size_t index_matche = 0;

        std::vector<Vector3d> list_points;

        std::vector<Vector3d> list_points_for_plan;

        float val_min = 100000000;
        float val_max = 0;

        for(stereo_match mt:  matchings_stereo)
        {
            int x = int(mt.first_point(1));
            int y = int(mt.first_point(0));
            float val_z = float(list_3d_points[index_matche](2));
                    if(double(val_z) > 0 && double(val_z) < 10000.0)
            {


                    myfile_3d_points << float(list_3d_points[index_matche](0)) << ";" << float(list_3d_points[index_matche](1)) << ";"  << float(list_3d_points[index_matche](2)) << "\n";

                                                                                               list_points.push_back(Vector3d(val_z,y,x));
                                              list_points_for_plan.push_back(Vector3d(float(list_3d_points[index_matche](2)),float(list_3d_points[index_matche](1)),float(list_3d_points[index_matche](0))));
        }
                                        index_matche++;
        }
                    myfile_3d_points.close();

            //if(index_img == 201)
            //    return;

            MatrixXf dst;
            //MatrixXf edges_left = mat_to_eigen(grad);
            MatrixXf edges_left = pcr.pc_first;


            MatrixXf left_eigen = mat_to_eigen(left_view);
            int th = otsu_segmentation(left_eigen);
            cout << "seuil " << th << endl;
            binarize_image(left_eigen,  dst , th , 0 , 255);
            Mat resl = eigen_to_mat(dst);

            string seg_image_name = string("..//SegResults//");
            seg_image_name.append(string("_")).append(to_string(index_img)).append(string(".png"));


            imwrite(seg_image_name,resl);

            typedef std::vector<Vector2i> markers;
            std::vector<markers> list_markers;

            Vector3i color_value;

            size_t index = 0;

            for(int row = 0 ; row < nrows; row++)
            {
                for(int col = 0; col < ncols; col++)
                {
                    if(int(dst(row,col))==255)
                    {
                        markers mark;
                        // take the point in format (y,x)
                        mark.push_back(Vector2i(row, col));
                        populate_marker_segmentation(mark, row, col, dst , index);
                        list_markers.push_back(mark);
                        index++;
                    }
                }
            }

            //cout << "Verification " << matchings_stereo

            typedef std::vector<Vector3d> _3d_points_segmented;
            std::vector<_3d_points_segmented> list_3d_points_segmented;

            std::vector<markers> list_correct_markers;


            for(int index_marker = 0 ; index_marker < list_markers.size() ; index_marker++)
            {
                markers mark = list_markers[index_marker];
                _3d_points_segmented temp_3d_points;
                for(int k = 0 ; k < mark.size() ; k++)
                {
                    Vector2i val_comp = mark[k];
                    for(int i = 0; i < list_points.size(); i ++)
                    {
                        Vector2i val = (list_points[i]).segment(1,2).cast<int>();
                        float norm = (val_comp - val).cast<float>().norm();
                        if(norm < 1)
                        {
                            temp_3d_points.push_back(list_points_for_plan[i]);
                        }
                    }
                }
                if(temp_3d_points.size() > 4)
                {
                    list_3d_points_segmented.push_back(temp_3d_points);
                    list_correct_markers.push_back(list_markers[index_marker]);
                }

            }

            list_markers = list_correct_markers;

            std::vector<float> mean_z(list_3d_points_segmented.size(), 0);

            std::vector<_3d_points_segmented> list_3d_points_segmented_wo_outliers;


            for (int index_3d = 0 ; index_3d < list_3d_points_segmented.size(); index_3d++) {
                cout << "objet numero " << index_3d << "    " << list_3d_points_segmented[index_3d].size() << endl;
                if(list_3d_points_segmented[index_3d].size() > 4)
                {
                    _3d_points_segmented temp_vect = list_3d_points_segmented[index_3d];
                    std::sort(begin(temp_vect), end(temp_vect), ascending_depth());
                    list_3d_points_segmented[index_3d] = temp_vect;
                    VectorXf z_points(temp_vect.size());
                    for(size_t i = 0 ; i < temp_vect.size() ; i++)
                    {
                        z_points[i] = temp_vect[i].z();
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
                    _3d_points_segmented vect_wo_outliers;

                    for(size_t i = 0 ; i < z_points.rows() ; i++  )
                    {
                        if(z_points[i] > lower_bound &&  z_points[i] < upper_bound)
                        {
                            inliers[n_inliers] = z_points[i];
                            inlier_indexes.push_back(i);
                            n_inliers ++;
                            //((list_3d_points_segmented[index_3d])[i])[2] = 0;
                            vect_wo_outliers.push_back((list_3d_points_segmented[index_3d])[i]);
                        }
                    }

                    list_3d_points_segmented_wo_outliers.push_back(vect_wo_outliers);


                    inliers = inliers.segment(0, n_inliers);

                    float mean_ = inliers.mean();

                    mean_z[index_3d] = mean_;

                }
            }


            std::string save_3d_points_segmanted_plane = dir_name;
            save_3d_points_segmanted_plane.append(string("segmented_plane_")).append(to_string(index_img)).append(".txt");
            ofstream myfile_3d_points_segmented_plane (save_3d_points_segmanted_plane);

            if (!myfile_3d_points_segmented_plane.is_open())
                continue;


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



                    for(int idx = 0 ; idx < list_3d_points_segmented_wo_outliers[index_marker].size(); idx++)
                    {

                        if( min_x > list_3d_points_segmented_wo_outliers[index_marker][idx][2] )
                            min_x = list_3d_points_segmented_wo_outliers[index_marker][idx][2];


                        if( max_x < list_3d_points_segmented_wo_outliers[index_marker][idx][2] )
                            max_x = list_3d_points_segmented_wo_outliers[index_marker][idx][2];


                        if( min_y > list_3d_points_segmented_wo_outliers[index_marker][idx][1] )
                            min_y = list_3d_points_segmented_wo_outliers[index_marker][idx][1];


                        if( max_y < list_3d_points_segmented_wo_outliers[index_marker][idx][1] )
                            max_y = list_3d_points_segmented_wo_outliers[index_marker][idx][1];

                    }

                    int cur = 0;

                    std::vector<Point3f> object_points;

                    for(int idx_x = int(min_x); idx_x < int(max_x); idx_x = idx_x + densite)
                    {
                        for(int idx_y = int(min_y); idx_y < int(max_y); idx_y = idx_y + densite)
                        {
                            //stereo_par
                            //cout << "un rectangle de   x=(" <<  min_x << " ; " << max_x << ")     y =(" << min_y
                            //     << " ; " << max_y << ")      idx_x = " << idx_x  << "  idx_y = "  << idx_y << "             " << cur << endl;
                            //cout << cur << "   " << ((int(max_x) - int(min_x) )/ densite) * ((int(max_y) - int(min_y) )/ densite)   << endl;
                            object_points.push_back(  Point3f(idx_x,idx_y,value) );
                            cur ++;
                        }
                    }

                    //cout << endl << endl;

                    cout << "taille 3D object " << object_points.size() << endl;

                    Mat rvec = stereo_par.R;
                    Mat tvec = stereo_par.T;
                    Mat camera_matrix = stereo_par.M1;
                    Mat distCoeffs = stereo_par.D1;
                    std::vector<Point2f> imagePoints;

                    cv::projectPoints(object_points,rvec,tvec,camera_matrix,distCoeffs,imagePoints);

                    cout << "un rectangle de x " <<  min_x << " ; " << max_x << "     y " << min_y << " ; " << max_y << endl;

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

                    if( nb_okay >  list_3d_points_segmented_wo_outliers[index_marker].size())
                    {
                        for(size_t id_3d_pt = 0 ; id_3d_pt < new_list_3d_object.size(); id_3d_pt++)
                        {
                            myfile_3d_points_segmented_plane << std::fixed << std::setprecision(5) << index_marker << ";"
                                                             << list_3d_points_segmented_wo_outliers.size() << ";"  << id_3d_pt << ";"
                                                             <<  nb_okay << ";"
                                                              << new_list_3d_object[id_3d_pt].x() << ";"
                                                              << new_list_3d_object[id_3d_pt].y() << ";"
                                                              << new_list_3d_object[id_3d_pt].z() << "\n";

                        }
                    }
                    else
                    {
                        for(int idx = 0 ; idx < list_3d_points_segmented_wo_outliers[index_marker].size(); idx++)
                        {

                            myfile_3d_points_segmented_plane << std::fixed << std::setprecision(5) << index_marker << ";"
                                                             << list_3d_points_segmented_wo_outliers.size() << ";"  << idx << ";"
                                                             << list_3d_points_segmented_wo_outliers[index_marker].size() << ";"
                                                             << list_3d_points_segmented_wo_outliers[index_marker][idx].x() << ";"
                                                             << list_3d_points_segmented_wo_outliers[index_marker][idx].y() << ";"
                                                             << list_3d_points_segmented_wo_outliers[index_marker][idx].z() << "\n";

                        }

                    }



                    cout << "ajout " << nb_okay << "  " << list_3d_points_segmented_wo_outliers[index_marker].size() << endl << endl << endl;
                    /**/
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
                        int row = mark[i].x();
                        int col = mark[i].y();
                        if(value > 0)
                            results_3d(row, col) = value;
                    }



                    for(int idx = 0 ; idx < list_3d_points_segmented_wo_outliers[index_marker].size(); idx++)
                    {

                        myfile_3d_points_segmented_plane << std::fixed << std::setprecision(5) << index_marker << ";"
                                                         << list_3d_points_segmented_wo_outliers.size() << ";"  << idx << ";"
                                                         << list_3d_points_segmented_wo_outliers[index_marker].size() << ";"
                                                         << list_3d_points_segmented_wo_outliers[index_marker][idx].x() << ";"
                                                         << list_3d_points_segmented_wo_outliers[index_marker][idx].y() << ";"
                                                         << list_3d_points_segmented_wo_outliers[index_marker][idx].z() << "\n";

                    }
                }
            }

            /*



                myfile_3d_points_segmented_plane.close();




                std::string save_3d_points_segmanted = dir_name;
                save_3d_points_segmanted.append(string("segmented_")).append(to_string(index_img)).append(".txt");
                ofstream myfile_3d_points_segmented (save_3d_points_segmanted);

                if (!myfile_3d_points_segmented.is_open())
                    continue;



                MatrixXf::Index maxRow_r, maxCol_r;
                float max_value = results_3d.maxCoeff(&maxRow_r, &maxCol_r);


                for(int row = 0 ; row < nrows; row++)
                {
                    for(int col = 0; col < ncols ; col ++)
                    {
                        if(double(results_3d(row, col)) != 0.0)
                        {
                            myfile_3d_points_segmented << std::fixed << std::setprecision(5) << row << ";" << col << ";"  << results_3d(row, col) << "\n";
                        }
                        else
                        {
                            myfile_3d_points_segmented << std::fixed << std::setprecision(5) << row << ";" << col << ";"  << 1.1*max_value  << "\n";
                            results_3d(row, col) = 1.1*15000;
                        }
                    }
                }

                myfile_3d_points_segmented.close();



                normalize_matrix(results_3d, 1);

                Matrix<Vector3d, Dynamic, Dynamic> colored_img(nrows, ncols);

                Matrix<Vector3d, Dynamic, Dynamic> blent_img(nrows, ncols);

                float alpha = 0.1;
                float beta;

                beta = ( 1.0 - alpha );

                for(int row = 0 ; row < nrows; row++)
                {
                    for(int col = 0; col < ncols ; col ++)
                    {
                        float r, g , b;
                        colormap<float>( ColorMapType::COLOR_MAP_TYPE_VIRIDIS, results_3d(row , col), r, g, b  );
                        colored_img(row,col) = 255.0 * Vector3d(r, g, b);
                        float new_red = alpha * left_eigen(row, col) + beta * 255 * r;
                        float new_green = alpha * left_eigen(row, col) + beta * 255 * g;
                        float new_blue = alpha * left_eigen(row, col) + beta * 255 * b;

                        blent_img(row, col) = Vector3d(new_red, new_green, new_blue);
                    }
                }

                Mat results_3d_open_cv =  eigen_to_mat_template<Vector3d>(blent_img);


                string image_name = string("..//Results//");
                image_name.append(string("_")).append(to_string(index_img)).append(string(".png"));

                imwrite(image_name,results_3d_open_cv);

                //system("pause");

                matchings_stereo.clear();
                /**/


        }
    }

    /**/
}

