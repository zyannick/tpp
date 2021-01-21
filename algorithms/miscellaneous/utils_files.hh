#pragma once

#include <iostream>
#include <string.h>
#include <cctype>
#include <stdio.h>
#include <time.h>
#include <ctime>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <set>
#include <list>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include "managed_path.hh"

#ifdef _WIN32
#include "dirent_win.h"
#endif

#ifdef linux
#include <dirent.h>
#endif
//#include "global_var.hh"

 
using namespace std;

namespace tpp
{

string miscellaneous;
	/// Permet de lire la liste d'images dans le fichier (fonctionne pour JSON).
	/// param filename: le nom du fichier, y compris son extension.
	/// param l: la liste dans laquelle on souhaite ajouter les noms des images.
    static bool readStringList(const string& filename, std::vector<string>& l)
	{
		l.resize(0);
		cv::FileStorage fs(filename, cv::FileStorage::READ);
		if (!fs.isOpened())
		{
			fs.release();
			cerr << "Cannot open file" << endl;
			return false;
		}
		cv::FileNode n = fs.getFirstTopLevelNode();
		if (n.type() != cv::FileNode::SEQ)
		{
			fs.release();
			cerr << "Cannot read file" << endl;
			return false;
		}
		cv::FileNodeIterator it = n.begin(), it_end = n.end();
		for (; it != it_end; ++it)
			l.push_back((string)*it);
		fs.release();
		cout << "JSON traité avec succes" << endl;
		return true;
	}

	///Trouve le seuil pour lequel on a le bon nombre de régions.
	/// param ImageIn : image 8 bits en niveaux de gris.
	/// param nbRegionsAttendues : le nombre de regions attendus suite au seuillage.
	/// return : le seuil maximal pour lequel on a le nombre de région attendues si trouvé. un nombre negatif sinon.
	static int rechercheSeuilOptimal(const cv::Mat &ImageIn, const int& nbRegionsAttendues)
	{
		cv::Mat img;
		cv::Mat labels, stats;
		cv::Mat centroids;
		for (int seuil = 254; seuil > 0; --seuil)
		{
			cv::threshold(ImageIn, img, seuil, 255, cv::THRESH_BINARY);
			cv::connectedComponentsWithStats(img, labels, stats, centroids, 8);
			if (centroids.rows - 1 == nbRegionsAttendues)
			{
				//printf("\n%s%d\n", "Seuil = ", seuil);
				return seuil;
			}
		}
		return -1;
	}

	/// Remplit les listes a partir du fichier d'entree
    static void remplirListes(const std::string& inputFilename, std::vector<std::string> &imageListCam1, std::vector<std::string> &imageListCam2)
	{
		imageListCam1.resize(0);
		imageListCam2.resize(0);
        std::vector<string> l;
		readStringList(inputFilename, l);
		std::string t1, t2;
		for (int i = 0; i < l.size(); i++)
		{
            t1 = miscellaneous;
            t2 = miscellaneous;
            t1 = t1.append( std::string("./Images/Camera1/"));
            t2 = t2.append(std::string("./Images/Camera2/"));
			t1.append(l[i]);
			t2.append(l[i]);
            cout << "fichier cam1 = " << t1 << "\nfichier cam2 = " << t2 << "\n\n" << endl;
			//cout << "fichier cam1 = " << t1 << "\nfichier cam2 = " << t2 << "\n\n" << endl;
			imageListCam1.push_back(t1);
			imageListCam2.push_back(t2);
		}
	}
    
    struct sort_string_descending
    {
        inline bool operator() (const string& s1, const string& s2)
        {
            return (s1.compare(s2) < 0);
        }
    };

    std::vector<string> get_list_of_files(string dir_name)
    {
        std::vector<string> list_dir;
        std::string parent_dir ("..");
        std::string current_dir (".");
        int evr = 1;
        DIR *dir;
        struct dirent *ent;
        int dir_cp = 1;
        int num_img = 0;
        
        if ((dir = opendir (dir_name.c_str())) != NULL)
        {
            //cout << "Dossier ouvert " << dir_name << endl;
            while ((ent = readdir (dir)) != NULL)
            {
                string file_name(ent->d_name);
                string path_to_file = dir_name;
                if(file_name.compare(parent_dir) != 0 && file_name.compare(current_dir) != 0)
                {
                    path_to_file.append("//").append(file_name);
                    list_dir.push_back(path_to_file);
                }
            }
            dir_cp++;
        }
        else
        {
            cout << "Impossible d'ouvrir le dossier" << dir_name << endl;
        }
        //cout << "get_list_of_files " << list_dir.size() << endl;
        closedir (dir);
        std::sort(list_dir.begin(),list_dir.end(),sort_string_descending());
        return list_dir;
    }



    /// Remplit les listes a partir du fichier d'entree
    static void remplir_listes_from_dir(const string path_to, std::vector<string> &imageListCam1, std::vector<string> &imageListCam2)
    {
        imageListCam1.resize(0);
        imageListCam2.resize(0);
        std::vector<string> l;
        std::string t1, t2;
        for (int i = 0; i < l.size(); i++)
        {
            t1 = path_to;
            t2 = path_to;
            t1 = t1.append( std::string("./Images/Camera1/"));
            t2 = t2.append(std::string("./Images/Camera2/"));
            t1.append(l[i]);
            t2.append(l[i]);
            cout << "fichier cam1 = " << t1 << "\nfichier cam2 = " << t2 << "\n\n" << endl;
            //cout << "fichier cam1 = " << t1 << "\nfichier cam2 = " << t2 << "\n\n" << endl;
            imageListCam1.push_back(t1);
            imageListCam2.push_back(t2);
        }
    }

    static void remplirListes(const string& inputFilename, std::vector<string> &imageListCam1, std::vector<string> &imageListCam2,
                              std::string preffix)
    {
        imageListCam1.resize(0);
        imageListCam2.resize(0);
        std::vector<string> l;
        readStringList(inputFilename, l);
        std::string t1, t2;

        for (int i = 0; i < l.size(); i++)
        {
            std::string p1 = miscellaneous;
            std::string p2 = miscellaneous;
            p1 = p1.append(preffix);
            p2 = p2.append(preffix);
            t1 = p1.append( std::string("Camera1/") );
            t2 = p2.append( std::string("Camera2/") );

            t1.append(l[i]);
            t2.append(l[i]);
            //cout << "fichier cam1 = " << t1 << "\nfichier cam2 = " << t2 << "\n\n" << endl;
            imageListCam1.push_back(t1);
            imageListCam2.push_back(t2);
        }
    }


    static void remplirListes_absolute(const string& inputFilename, std::vector<string> &imageListCam1, std::vector<string> &imageListCam2,
                              std::string preffix)
    {
        imageListCam1.resize(0);
        imageListCam2.resize(0);
        std::vector<string> l;
        readStringList(inputFilename, l);
        std::string t1, t2;

        for (int i = 0; i < l.size(); i++)
        {
            std::string p1 = miscellaneous;
            std::string p2 = miscellaneous;
            p1 = p1.append(preffix);
            p2 = p2.append(preffix);

            t1 = p1;
            t2 = p2;

            t1.append(l[i]);
            t2.append(l[i]);
            //cout << "fichier cam1 = " << t1 << "\nfichier cam2 = " << t2 << "\n\n" << endl;
            imageListCam1.push_back(t1);
            imageListCam2.push_back(t2);
        }
    }

	/// Remplit les listes a partir du fichier d'entree
    static void remplirListes(const string& inputFilename, std::vector<string> &imageListCam1)
	{
		imageListCam1.resize(0);
        std::vector<string> l;
		readStringList(inputFilename, l);
		std::string t1, t2;
		for (int i = 0; i < l.size(); i++)
		{
            t1 = miscellaneous;
            t1 = t1.append( std::string("./Images/Camera1/"));
			t1.append(l[i]);
			//cout << "fichier cam1 = " << t1 << "\nfichier cam2 = " << t2 << "\n\n" << endl;
			imageListCam1.push_back(t1);
		}
	}

	/// Remplit les listes a partir du fichier d'entree
    static void remplirListes(const string& inputFilename, std::vector<string> &imageListCam1, int side)
	{
		imageListCam1.resize(0);
        std::vector<string> l;
		readStringList(inputFilename, l);
		std::string t1, t2;
		for (int i = 0; i < l.size(); i++)
		{
            t1 = miscellaneous;

			if (side == 0)
                t1 = t1.append(std::string("./imagecorrect/C1/"));
			else
                t1 = t1.append(std::string("./imagecorrect/C2/"));
			t1.append(l[i]);
            //cout << "fichier cam1 = " << t1 << "\nfichier cam2 = " << t2 << "\n\n" << endl;
			imageListCam1.push_back(t1);
		}
	}

	///export CSV : permet d'exporter les points des listes en .csv.
	/// fonction pour du debug.
	/// Ne pas oublier d'ouvrir le fichier avant la premier appel et de le fermer après le dernier appel de la fonction.
    static void exportCSV(std::ofstream& myfile, const std::vector<cv::Point2f> &imagePoints1, const std::vector<cv::Point2f> &imagePoints2)
	{
		std::string separator = std::string(";");

		//myfile << "x1" << separator << "y1" << separator << "x2" << separator << "y2" << separator << endl;
		for (auto i = 0; i < imagePoints1.size(); ++i)
		{
			myfile << imagePoints1[i].x << separator << imagePoints1[i].y << separator;
		}
		myfile << "|" << separator;

		for (auto i = 0; i < imagePoints2.size(); ++i)
		{
			myfile << imagePoints2[i].x << separator << imagePoints2[i].y << separator;
		}

		myfile << endl;
	}

	//banks
	inline cv::Mat from_vector_to_matrix(std::vector<cv::Point2f> image_points)
	{
		//cout << endl << endl << "from_vector_to_matrix" << endl << endl;
		int N = image_points.size();
		cv::Mat campnts(1, N, CV_64FC2);
		for (auto i = 0; i < N; i++)
		{
			double val = (image_points[i]).x;
			campnts.at<cv::Vec2d>(0, i)[0] = val;
			val = (image_points[i]).y;
			campnts.at<cv::Vec2d>(0, i)[1] = val;
			//cout << "val " << campnts.at<Vec2d>(0, i) << endl;
		}
		//cout << "end kkikik" << endl;
		return campnts;
	}

	inline void from_vector_to_matrix(std::vector<cv::Point2f> image_points, cv::Mat &campnts)
	{
		//cout << endl << endl << "from_vector_to_matrix" << endl << endl;
		int N = image_points.size();
		for (auto i = 0; i < N; i++)
		{
			double val = (image_points[i]).x;
			campnts.at<cv::Vec2d>(0, i)[0] = val;
			val = (image_points[i]).y;
			campnts.at<cv::Vec2d>(0, i)[1] = val;
			//cout << "val " << campnts.at<Vec2d>(0, i) << endl;
		}
		//cout << "end kkikik" << endl;
	}
}
