#pragma once
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include "calibration_and_3d/calibration/define_calibrate.hh"
#include "algorithms/miscellaneous.hh"
#include "calibration_and_3d/calibration/extract_points.hh"

using namespace std;
 

namespace tpp
{
/**
     * @brief dist_between_points
     * @param pt1
     * @param pt2
     * @return
     */
inline double dist_between_points(cv::Point2f pt1, cv::Point2f pt2)
{
    double the_norm = sqrt(pow(pt1.x - pt2.x, 2) + pow(pt1.y - pt2.y, 2));
    return the_norm;
}

/**
     * @brief getCentroid
     * @param img
     * @return
     */
cv::Point2f getCentroid(cv::Mat img)
{
    cv::Point2f Coord;
    cv::Moments mm = cv::moments(img, false);
    double moment10 = mm.m10;
    double moment01 = mm.m01;
    double moment00 = mm.m00;
    Coord.x = moment10 / moment00;
    Coord.y = moment01 / moment00;
    return Coord;
}


cv::Point2f getMarkedPoint(cv::Mat img)
{
    imwrite("pom.bmp",img);
    cv::Point2f coord;
    for(auto row = 0 ; row < img.rows ; row++)
    {
        auto col = 0;
        for(col = 0 ; col < img.cols ; col++)
        {
            cv::Vec3b val = img.at<cv::Vec3b>(row,col);
            //cout << "val " << val << endl;
            if(val[0] != val[1] && val[1]  != val[2] )
            {
                coord.x = col;
                coord.y = row;
                break;
            }
        }
    }
    return coord;
}




/**
     * @brief extract_object_point
     * @param object_points_left
     * @param object_points_right
     * @param img_left
     * @param img_right
     * @param point_feature
     */
inline
void extract_object_point(std::vector<cv::Point2f> & object_points_left,
                          std::vector<cv::Point2f>  &object_points_right, cv::Mat img_left, cv::Mat img_right, OBJECT_POINT_FEATURE point_feature)
{
    if (OBJECT_POINT_FEATURE::GRAVITY_CENTER == point_feature)
    {
        std::vector<cv::Point2f> im_point_left;
        im_point_left.push_back(getCentroid(img_left));
        object_points_left = im_point_left;
        std::vector<cv::Point2f> im_point_right;
        im_point_right.push_back(getCentroid(img_right));
        object_points_right = im_point_right;
    }
}

cv::Mat convert_16_to_8(string file_name)
{
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::Mat view = cv::imread(file_name, CV_16U);
    cv::Mat view_8bits;
    cv::minMaxLoc(view, &minVal, &maxVal, &minLoc, &maxLoc);
    view.convertTo(view_8bits, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
    return view_8bits;
}


void augment_data_around_point(cv::Point2f mark_l_gc, cv::Point2f mark_r_gc, int data_augment, std::vector<cv::Point2f>& object_points_left,
                               std::vector<cv::Point2f> & object_points_right)
{

    int w = 1;
    int nimages = 1;
    int nb = (w*2)+1;

    int nb_points;

    mark_l_gc = cv::Point2f( int(mark_l_gc.x) , int(mark_l_gc.y));
    mark_r_gc = cv::Point2f( int(mark_r_gc.x) , int(mark_r_gc.y));


    //Pas d'augmentation de données
    if(data_augment == 0)
        nb_points = nimages;
    //On augmente en faisant une bijection sur les 9 voisins
    else if(data_augment == 1)
        nb_points = nimages*nb*nb;
    //On augment en variant sur les voisins
    else if (data_augment == 2)
        nb_points = nimages*nb*nb;
    //On fait un combinaison sur les 9 voisins : 81 combinaisons
    else if (data_augment == 3)
        nb_points = nimages*nb*nb*nb*nb;

    object_points_left.clear();
    object_points_right.clear();
    object_points_left.resize(nb_points);
    object_points_right.resize(nb_points);

    int j = 0;

    if(data_augment == 0)
    {
        cv::Point2f left_pt = mark_l_gc /*+ Point2f(col,row)*/;
        //vector<Point2f> im_point_left;
        //im_point_left.push_back(left_pt);
        object_points_left[j] = left_pt;
        cout << "mark_l_gc " << mark_l_gc << " left_pt " << left_pt << endl;

        cv::Point2f right_pt = mark_r_gc /* + Point2f(col,row)*/;
        //vector<Point2f> im_point_right;
        //im_point_right.push_back(right_pt);
        object_points_right[j] = right_pt;
        cout << "mark_r_gc " << mark_r_gc << " right_pt " << right_pt << endl;
        j++;
    }
    else if (data_augment == 1) {
        for(int row = -w ; row <= w ; row++)
        {
            for(int col = -w; col <= w ; col++)
            {
                cv::Point2f left_pt = mark_l_gc + cv::Point2f(col,row);
                //vector<Point2f> im_point_left;
                //im_point_left.push_back(left_pt);
                object_points_left[j] = left_pt;
                cout << "mark_l_gc " << mark_l_gc << " left_pt " << left_pt << endl;

                cv::Point2f right_pt = mark_r_gc + cv::Point2f(col,row);
                //vector<Point2f> im_point_right;
                //im_point_right.push_back(right_pt);
                object_points_right[j] = right_pt;
                cout << "mark_r_gc " << mark_r_gc << " right_pt " << right_pt << endl;
                cout << "valeur j " << j << endl;
                j++;
            }
        }
    }
    else if (data_augment == 2) {
        for(int row = -w ; row <= w ; row++)
        {
            for(int col = -w; col <= w ; col++)
            {
                cv::Point2f left_pt = mark_l_gc;
                //vector<Point2f> im_point_left;
                //im_point_left.push_back(left_pt);
                object_points_left[j] = left_pt;
                cout << "mark_l_gc " << mark_l_gc << " left_pt " << left_pt << endl;

                cv::Point2f right_pt = mark_r_gc + cv::Point2f(col,row);
                //vector<Point2f> im_point_right;
                //im_point_right.push_back(right_pt);
                object_points_right[j] = right_pt;
                cout << "mark_r_gc " << mark_r_gc << " right_pt " << right_pt << endl;
                cout << "valeur j " << j << endl;
                j++;
            }
        }
    }
    else if (data_augment == 3) {
        for(int row = -w ; row <= w ; row++)
        {
            for(int col = -w; col <= w ; col++)
            {
                for(int rw = -w ; rw <= w ; rw++)
                {
                    for(int cl = -w; cl <= w ; cl++)
                    {
                        cv::Point2f left_pt = mark_l_gc + cv::Point2f(col,row);
                        //vector<Point2f> im_point_left;
                        //im_point_left.push_back(left_pt);
                        object_points_left[j] = left_pt;
                        //cout << "mark_l_gc " << mark_l_gc << " left_pt " << left_pt << endl;

                        cv::Point2f right_pt = mark_r_gc + cv::Point2f(cl,rw);
                        //vector<Point2f> im_point_right;
                        //im_point_right.push_back(right_pt);
                        object_points_right[j] = right_pt;
                        cout << "left  " << left_pt  << "              right " << right_pt << endl;
                        cout << "valeur j " << j << endl;
                        j++;
                    }
                }
            }
        }
    }

}

/**
     * @brief plane_reconstruction_marked
     * @param mode
     * @param points_plan_left
     * @param points_plan_right
     * @param imageSize
     * @param inputFile
     */
void plane_reconstruction_marked(Mode_To_Revtreive_Files mode, std::vector< std::vector<cv::Point2f> >& object_points_left,
                                 std::vector< std::vector<cv::Point2f> >& object_points_right, int &nimages,bool is_ground, bool only_marked,
                                 string file_name="", string file_name_marked="")
{
    std::vector<string> list_left, list_right;
    std::vector<string> list_left_marked, list_right_marked;
    int wind = 3;
    //If it is the plane of the ground
    string input_file_name;
    string input_file_name_marked;
    string preffix = std::string("./Images/Table/Normal/");
    string preffix_marked = std::string("./Images/Table/Marked/");
    if (mode == DIRECTORY)
    {
        //getListFilesOfDirectory(imageListCam1, imageListCam2);
    }
    else if (mode == JSFILES)
    {
        if (file_name.length() == 0)
        {
            if(!only_marked)
                input_file_name = std::string("./json/table_troyes.json");// fichier par défaut
            input_file_name_marked = std::string("./json/table_troyes_marked.json");// fichier par défaut
        }
        else
        {
            input_file_name = file_name;

        }
    }

    // extrait les noms des images a partir du fichier json
    if(is_ground)
    {
        if(!only_marked)
            remplirListes(input_file_name, list_left, list_right);
        remplirListes(input_file_name_marked, list_left_marked, list_right_marked);
    }
    else
    {
        if(!only_marked)
            remplirListes(input_file_name, list_left, list_right,preffix);
        remplirListes(input_file_name_marked, list_left_marked, list_right_marked,preffix_marked);
    }

    if(!only_marked)
        nimages = list_left.size();
    else
        nimages = list_left_marked.size();


    int data_augment = 0;


    assert(nimages > 0 && "You need at least one image pair");

    if(!only_marked)
        assert(list_left.size() == list_right.size() && "The number of images are not the same");
    else
        assert(list_left_marked.size() == list_right_marked.size() && "The number of images are not the same");

    int w = 1;

    int nb = (w*2)+1;

    int nb_points;

    //Pas d'augmentation de données
    if(data_augment == 0)
        nb_points = nimages;
    //On augmente en faisant une bijection sur les 9 voisins
    else if(data_augment == 1)
        nb_points = nimages*nb*nb;
    //On augment en variant sur les voisins
    else if (data_augment == 2)
        nb_points = nimages*nb*nb;
    //On fait un combinaison sur les 9 voisins : 81 combinaisons
    else if (data_augment == 3)
        nb_points = nimages*nb*nb*nb*nb;



    object_points_left.clear();
    object_points_right.clear();
    object_points_left.resize(nb_points);
    object_points_right.resize(nb_points);

    int j = 0;

    for (int i = 0; i < nimages; i++)
    {
        cv::Mat view_left_8bits;
        cv::Mat view_right_8bits;

        if(!only_marked)
        {
            view_left_8bits = convert_16_to_8(list_left[i]);
            view_right_8bits = convert_16_to_8(list_right[i]);
        }

        cv::Mat view_left_8bits_marked = cv::imread(list_left_marked[i], cv::IMREAD_COLOR);
        cv::Mat view_right_8bits_marked = cv::imread(list_right_marked[i], cv::IMREAD_COLOR);

        cv::Point2f mark_l =  getMarkedPoint(view_left_8bits_marked);
        cv::Point2f mark_l_gc;


        if(!only_marked)
            get_local_maxima_gc(wind, view_left_8bits, mark_l, mark_l_gc);
        else
            mark_l_gc = mark_l;

        //cout << "mark_l " << mark_l << " mark_l_gc " << mark_l_gc << endl;

        cv::Point2f mark_r =  getMarkedPoint(view_right_8bits_marked);
        cv::Point2f mark_r_gc;
        if(!only_marked)
            get_local_maxima_gc(wind, view_right_8bits, mark_r, mark_r_gc);
        else
            mark_r_gc = mark_r;

        if(data_augment == 0)
        {
            cv::Point2f left_pt = mark_l_gc /*+ Point2f(col,row)*/;
            std::vector<cv::Point2f> im_point_left;
            im_point_left.push_back(left_pt);
            object_points_left[j] = im_point_left;
            cout << "mark_l_gc " << mark_l_gc << " left_pt " << left_pt << endl;

            cv::Point2f right_pt = mark_r_gc /* + Point2f(col,row)*/;
            std::vector<cv::Point2f> im_point_right;
            im_point_right.push_back(right_pt);
            object_points_right[j] = im_point_right;
            cout << "mark_r_gc " << mark_r_gc << " right_pt " << right_pt << endl;
            j++;
        }
        else if (data_augment == 1) {
            for(int row = -w ; row <= w ; row++)
            {
                for(int col = -w; col <= w ; col++)
                {
                    cv::Point2f left_pt = mark_l_gc + cv::Point2f(col,row);
                    std::vector<cv::Point2f> im_point_left;
                    im_point_left.push_back(left_pt);
                    object_points_left[j] = im_point_left;
                    cout << "mark_l_gc " << mark_l_gc << " left_pt " << left_pt << endl;

                    cv::Point2f right_pt = mark_r_gc + cv::Point2f(col,row);
                    std::vector<cv::Point2f> im_point_right;
                    im_point_right.push_back(right_pt);
                    object_points_right[j] = im_point_right;
                    cout << "mark_r_gc " << mark_r_gc << " right_pt " << right_pt << endl;
                    cout << "valeur j " << j << endl;
                    j++;
                }
            }
        }
        else if (data_augment == 2) {
            for(int row = -w ; row <= w ; row++)
            {
                for(int col = -w; col <= w ; col++)
                {
                    cv::Point2f left_pt = mark_l_gc;
                    std::vector<cv::Point2f> im_point_left;
                    im_point_left.push_back(left_pt);
                    object_points_left[j] = im_point_left;
                    cout << "mark_l_gc " << mark_l_gc << " left_pt " << left_pt << endl;

                    cv::Point2f right_pt = mark_r_gc + cv::Point2f(col,row);
                    std::vector<cv::Point2f> im_point_right;
                    im_point_right.push_back(right_pt);
                    object_points_right[j] = im_point_right;
                    cout << "mark_r_gc " << mark_r_gc << " right_pt " << right_pt << endl;
                    cout << "valeur j " << j << endl;
                    j++;
                }
            }
        }
        else if (data_augment == 3) {
            for(int row = -w ; row <= w ; row++)
            {
                for(int col = -w; col <= w ; col++)
                {
                    for(int rw = -w ; rw <= w ; rw++)
                    {
                        for(int cl = -w; cl <= w ; cl++)
                        {
                            cv::Point2f left_pt = mark_l_gc + cv::Point2f(col,row);
                            std::vector<cv::Point2f> im_point_left;
                            im_point_left.push_back(left_pt);
                            object_points_left[j] = im_point_left;
                            cout << "mark_l_gc " << mark_l_gc << " left_pt " << left_pt << endl;

                            cv::Point2f right_pt = mark_r_gc + cv::Point2f(cl,rw);
                            std::vector<cv::Point2f> im_point_right;
                            im_point_right.push_back(right_pt);
                            object_points_right[j] = im_point_right;
                            cout << "mark_r_gc " << mark_r_gc << " right_pt " << right_pt << endl;
                            cout << "valeur j " << j << endl;
                            j++;
                        }
                    }
                }
            }
        }



        cout << endl << endl << endl;

    }

    cout << "here " << endl;
}

/**
     * @brief plane_reconstruction
     * @param mode
     * @param points_plan_left
     * @param points_plan_right
     * @param imageSize
     * @param inputFile
     */
void plane_reconstruction(Mode_To_Revtreive_Files mode,
                          std::vector<std::vector<cv::Point2f> > &points_plan_left,
                          std::vector<std::vector<cv::Point2f> > &points_plan_right, cv::Size &imageSize, string inputFile = "")
{
    std::vector<string> imageListCam1, imageListCam2;
    string inputFilename;
    if (mode == DIRECTORY)
    {
        //getListFilesOfDirectory(imageListCam1, imageListCam2);
    }
    else if (mode == JSFILES)
    {
        if (inputFile.length() == 0)
        {
            inputFilename = std::string("floor.json");// fichier par défaut
        }
        else
            inputFilename = inputFile;
    }
    cv::Size boardSize;
    int i, j, k, nimages;

    boardSize.width = 6;
    boardSize.height = 6;

    TYPE_PLAN type_plan = TYPE_PLAN::HORIZONTAL;

    // extrait les noms des images a partir du fichier json
    remplirListes(inputFilename, imageListCam1, imageListCam2);
    nimages = imageListCam1.size();

    assert(nimages == 1 && "You need only one image pair");

    assert(imageListCam1.size() == imageListCam2.size() && "The number of images are not the same");

    bool extrac_ok;
    int number_points = boardSize.height*boardSize.width;
    int nb_ok = 0;

    points_plan_left.clear();
    points_plan_right.clear();
    points_plan_left.resize(1);
    points_plan_right.resize(1);

    int idx_ok = 0;

    for (int i = 0; i < (int)imageListCam1.size(); i++)
    {
        cv::Mat view = cv::imread(imageListCam1[i], CV_16U);
        std::vector<cv::Point2f> im_point1;
        extrairePointsImage(view, boardSize, im_point1, imageListCam1[i], type_plan);
        cv::Mat view1 = cv::imread(imageListCam2[i], CV_16U);
        std::vector<cv::Point2f> im_point2;
        extrairePointsImage(view1, boardSize, im_point2, imageListCam2[i], type_plan);
        std::vector<cv::Point2f> im_point_matches;
        im_point_matches.resize(im_point1.size());

        for (int i = 0; i < im_point1.size(); i++)
        {
            double min_norm = 10000;
            int ind_min;
            for (int j = 0; j < im_point2.size(); j++)
            {
                double dist = dist_between_points(im_point1[i], im_point2[j]);
                if (min_norm > dist)
                {
                    min_norm = dist;
                    ind_min = j;
                }
            }
            im_point_matches[i] = im_point2[ind_min];
        }
        im_point2 = im_point_matches;

        imageSize = view.size();
        cv::Mat img_l = cv::Mat::zeros(imageSize, CV_8UC3);
        for (int pt = 0; pt < im_point1.size(); pt++)
        {
            if (pt % 3 == 0)
                img_l.at<cv::Vec3b>(im_point1[pt])[0] = 255;
            else if (pt % 3 == 1)
                img_l.at<cv::Vec3b>(im_point1[pt])[1] = 255;
            else if (pt % 3 == 2)
                img_l.at<cv::Vec3b>(im_point1[pt])[2] = 255;
        }
        cv::imwrite("tessst_l.bmp", img_l);

        cv::Mat img_r = cv::Mat::zeros(imageSize, CV_8UC3);
        for (int pt = 0; pt < im_point1.size(); pt++)
        {
            if (pt % 3 == 0)
                img_r.at<cv::Vec3b>(im_point2[pt])[0] = 255;
            else if (pt % 3 == 1)
                img_r.at<cv::Vec3b>(im_point2[pt])[1] = 255;
            else if (pt % 3 == 2)
                img_r.at<cv::Vec3b>(im_point2[pt])[2] = 255;
        }
        imwrite("tessst_r.bmp", img_r);

        points_plan_left[0] = im_point1;
        points_plan_right[0] = im_point2;
    }
}

/**
     * @brief plane_reconstruction
     * @param mode
     * @param object_points_left
     * @param object_points_right
     * @param point_feature
     * @param file_name
     */
void plane_reconstruction(Mode_To_Revtreive_Files mode, std::vector< std::vector<cv::Point2f> >& object_points_left,
                          std::vector< std::vector<cv::Point2f> >& object_points_right, OBJECT_POINT_FEATURE point_feature, string file_name = "")
{
    std::vector<string> imageListCam1, imageListCam2;
    int wind = 10;
    string inputFilename;
    if (mode == DIRECTORY)
    {
        //getListFilesOfDirectory(imageListCam1, imageListCam2);
    }
    else if (mode == JSFILES)
    {
        if (file_name.length() == 0)
        {
            inputFilename = std::string("file_object_plan.json");// fichier par défaut
        }
        else
            inputFilename = file_name;
    }
    int i, j, k;

    // extrait les noms des images a partir du fichier json
    remplirListes(inputFilename, imageListCam1, imageListCam2);
    int nimages = imageListCam1.size();

    assert(nimages > 0 && "You need at least one image pair");

    assert(imageListCam1.size() == imageListCam2.size() && "The number of images are not the same");

    bool extrac_ok;
    int nb_ok = 0;

    object_points_left.clear();
    object_points_right.clear();
    object_points_left.resize(1);
    object_points_right.resize(1);

    double minVal_left, maxVal_left;
    cv::Point minLoc_left, maxLoc_left;

    double minVal_right, maxVal_right;
    cv::Point minLoc_right, maxLoc_right;

    (object_points_left[0]).resize(nimages);
    (object_points_right[0]).resize(nimages);

    for (int i = 0; i < nimages; i++)
    {
        std::vector<cv::Point2f> im_point_left;
        std::vector<cv::Point2f> im_point_right;
        cv::Mat view_left = cv::imread(imageListCam1[i], CV_16U);
        cv::Mat view_left_8bits;
        cv::minMaxLoc(view_left, &minVal_left, &maxVal_left, &minLoc_left, &maxLoc_left);

        cv::Mat view_right = cv::imread(imageListCam2[i], CV_16U);
        cv::Mat view_right_8bits;
        cv::minMaxLoc(view_right, &minVal_right, &maxVal_right, &minLoc_right, &maxLoc_right);

        if (OBJECT_POINT_FEATURE::MAX_INTENSITY == point_feature)
        {
            im_point_left.push_back(cv::Point2f(maxLoc_left.x, maxLoc_left.y));
            im_point_right.push_back(cv::Point2f(maxLoc_right.x, maxLoc_right.y));
        }
        else
        {
            view_left.convertTo(view_left_8bits, CV_8U, 255.0 / (maxVal_left - minVal_left), -minVal_left * 255.0 / (maxVal_left - minVal_left));
            view_right.convertTo(view_right_8bits, CV_8U, 255.0 / (maxVal_right - minVal_right), -minVal_right * 255.0 / (maxVal_right - minVal_right));
            extract_object_point(im_point_left, im_point_right, view_left_8bits, view_right_8bits, point_feature);
        }
        (object_points_left[0])[i] = im_point_left[0];
        (object_points_right[0])[i] = im_point_right[0];
    }
}

/**
     * @brief get_object_points
     * @param mode
     * @param object_points_left
     * @param object_points_right
     * @param nimages
     * @param point_feature
     * @param file_name
     */
void get_object_points(Mode_To_Revtreive_Files mode, std::vector< std::vector<cv::Point2f> >& object_points_left,
                       std::vector< std::vector<cv::Point2f> >& object_points_right, int &nimages, OBJECT_POINT_FEATURE point_feature, string file_name)
{
    std::vector<string> imageListCam1, imageListCam2;
    int wind = 10;
    string inputFilename;
    if (mode == DIRECTORY)
    {
        //getListFilesOfDirectory(imageListCam1, imageListCam2);
    }
    else if (mode == JSFILES)
    {
        if (file_name.length() == 0)
        {
            inputFilename = std::string("file_object.json");// fichier par défaut
        }
        else
            inputFilename = file_name;
    }
    int i, j, k;

    // extrait les noms des images a partir du fichier json
    remplirListes(inputFilename, imageListCam1, imageListCam2);
    nimages = imageListCam1.size();

    assert(nimages > 0 && "You need at least one image pair");

    assert(imageListCam1.size() == imageListCam2.size() && "The number of images are not the same");

    bool extrac_ok;
    int nb_ok = 0;

    object_points_left.clear();
    object_points_right.clear();
    object_points_left.resize(nimages);
    object_points_right.resize(nimages);

    double minVal_left, maxVal_left;
    cv::Point minLoc_left, maxLoc_left;

    double minVal_right, maxVal_right;
    cv::Point minLoc_right, maxLoc_right;

    for (int i = 0; i < nimages; i++)
    {
        cv::Mat view_left = cv::imread(imageListCam1[i], CV_16U);
        cv::Mat view_left_8bits;
        cv::minMaxLoc(view_left, &minVal_left, &maxVal_left, &minLoc_left, &maxLoc_left);

        cv::Mat view_right = cv::imread(imageListCam2[i], CV_16U);
        cv::Mat view_right_8bits;
        cv::minMaxLoc(view_right, &minVal_right, &maxVal_right, &minLoc_right, &maxLoc_right);

        if (OBJECT_POINT_FEATURE::MAX_INTENSITY == point_feature)
        {
            std::vector<cv::Point2f> im_point_left;
            im_point_left.push_back(cv::Point2f(maxLoc_left.x, maxLoc_left.y));
            object_points_left[i] = im_point_left;

            std::vector<cv::Point2f> im_point_right;
            im_point_right.push_back(cv::Point2f(maxLoc_right.x, maxLoc_right.y));
            object_points_right[i] = im_point_right;
        }
        else
        {
            view_left.convertTo(view_left_8bits, CV_8U, 255.0 / (maxVal_left - minVal_left), -minVal_left * 255.0 / (maxVal_left - minVal_left));
            view_right.convertTo(view_right_8bits, CV_8U, 255.0 / (maxVal_right - minVal_right), -minVal_right * 255.0 / (maxVal_right - minVal_right));
            extract_object_point(object_points_left[i], object_points_right[i], view_left_8bits, view_right_8bits, point_feature);
        }
    }
}
}
