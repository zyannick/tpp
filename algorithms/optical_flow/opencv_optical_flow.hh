#pragma once


#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
//#include <opencv2/optflow.hpp>

#include "algorithms/miscellaneous.hh"
#include "global_utils.hh"
 
using namespace std;
using namespace tpp;




int open_cv_optical_flow(string path)
{

    std::vector<string> list_images;
    list_images = get_list_of_files(path);

    size_t nimages = list_images.size();

    Mat frame1, prvs, next;

    create_directory("..//tvl1_output_lepton2_x4");

    /*for(size_t index_img = 1 ;index_img < nimages ; index_img++)
    {
        cout << "image optical flow " << index_img << endl;

        prvs = read_16_converto_8(list_images[index_img-1]);
        next = read_16_converto_8(list_images[index_img]);

        //imwrite("..//lepton2//frame_" + to_string(100000 + index_img-1) + ".png" , prvs);
        //imwrite("..//lepton2//frame_" + to_string(100000 + index_img) + ".png" , next);


        Mat flow(prvs.size(), CV_32FC2);
        cv::Ptr<cv::superres::DualTVL1OpticalFlow> tv1 = cv::superres::createOptFlow_DualTVL1();
        tv1->calc(prvs, next, flow);
        //calcOpticalFlowFarneback(prvs, next, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
        //cout << "visualization" << endl;
        // visualization
        Mat flow_parts[2];
        split(flow, flow_parts);
        Mat magnitude, angle, magn_norm;
        cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
        normalize(magnitude, magn_norm, 0.0f, 1.0f, NORM_MINMAX);
        angle *= ((1.f / 360.f) * (180.f / 255.f));
        //cout << "build hsv image" << endl;
        //build hsv image
        Mat _hsv[3], hsv, hsv8, bgr;
        _hsv[0] = angle;
        _hsv[1] = Mat::ones(angle.size(), CV_32F);
        _hsv[2] = magn_norm;
        merge(_hsv, 3, hsv);
        hsv.convertTo(hsv8, CV_8U, 255.0);
        cvtColor(hsv8, bgr, COLOR_HSV2BGR);
        imwrite("..//tvl1_output_lepton2_x4//frame_"+ to_string(100000+index_img) + ".png", bgr);
    }*/
}


int generate_data_flow(string path)
{
    string output_path = "atrium_red_8";

    std::vector<string> list_images;
    list_images = get_list_of_files(path);

    size_t nimages = list_images.size();

    Mat frame1, frame2, frame3;

    create_directory("..//" + output_path);

    std::string all_temp = "..";
    all_temp = all_temp + "//"+ output_path +"_image_list.txt";
    ofstream all_log (all_temp,  std::ios_base::app);

    for(size_t index_img = 0 ;index_img < nimages-2 ; index_img++)
    {
        cout << "image optical flow " << index_img << endl;

        frame1 = read_16_converto_8(list_images[index_img]);

        string file1 = "frame_" + to_string(100000 + index_img) + ".png" ;
        string file2 = "frame_" + to_string(100000 + index_img + 1) + ".png" ;
        string file3 = "frame_" + to_string(100000 + index_img + 2) + ".png" ;
        all_log  << file1 << " " << file2 << " " << file3 << " " << to_string(100000 + index_img) << endl;


        imwrite("..//" + output_path + "//frame_" + to_string(100000 + index_img) + ".png" , frame1);
    }

    frame2 = read_16_converto_8(list_images[nimages - 2]);
    imwrite("..//" + output_path + "//frame_" + to_string(100000 + nimages - 2) + ".png" , frame2);
    frame3 = read_16_converto_8(list_images[nimages - 1]);
    imwrite("..//" + output_path + "//frame_" + to_string(100000 + nimages - 1) + ".png" , frame3);

    return 1;

}


int generate_data_flow(string output,string path, string prefix)
{
    string output_path = output;

    std::vector<string> list_images;
    list_images = get_list_of_files(path);

    size_t nimages = list_images.size();

    Mat frame1, frame2, frame3;

    create_directory("..//" + output_path);

    std::string all_temp = "..";
    all_temp = all_temp + "//"+ output_path +"_image_list.txt";
    ofstream all_log (all_temp,  std::ios_base::app);

    for(size_t index_img = 0 ;index_img < nimages-2 ; index_img++)
    {
       //cout << "image optical flow " << index_img << endl;

        frame1 = read_16_converto_8(list_images[index_img]);

        string file1 = "frame_" + prefix + "_" + to_string(100000 + index_img) + ".png" ;
        string file2 = "frame_" + prefix + "_" + to_string(100000 + index_img + 1) + ".png" ;
        string file3 = "frame_" + prefix + "_" + to_string(100000 + index_img + 2) + ".png" ;
        all_log  << file1 << " " << file2 << " " << file3 << " " << to_string(100000 + index_img) << endl;


        imwrite("..//" + output_path + "//frame_" + prefix + "_" + to_string(100000 + index_img) + ".png" , frame1);
    }

    frame2 = read_16_converto_8(list_images[nimages - 2]);
    imwrite("..//" + output_path + "//frame_"  + prefix + "_" +  to_string(100000 + nimages - 2) + ".png" , frame2);
    frame3 = read_16_converto_8(list_images[nimages - 1]);
    imwrite("..//" + output_path + "//frame_"  + prefix + "_" +  to_string(100000 + nimages - 1) + ".png" , frame3);

    return 1;

}
