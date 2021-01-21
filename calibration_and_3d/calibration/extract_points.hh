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

#include "algorithms/miscellaneous.hh"

#include "algorithms/interpolation/interpolation.hh"

using namespace std;

namespace tpp
{
	enum TYPE_PLAN { VERTICAL, HORIZONTAL };

	///Extrait les points de l'image correspondants à la grille.
    static bool extrairePointsImage(const cv::Mat &ImageIn, const cv::Size &boardSize,
        std::vector<cv::Point2f> &imagePoints, std::string &imageName,int wind = 3, TYPE_PLAN type_plan = TYPE_PLAN::VERTICAL)
	{
		cv::Mat img;
		cv::Mat img_8bit;
		bool res = true;
		// Conversion en 8bit
		double minVal, maxVal;
		cv::Point minLoc, maxLoc;
		std::string NomFichier;
		cv::minMaxLoc(ImageIn, &minVal, &maxVal, &minLoc, &maxLoc);
		//Conversion de l'image en 8 bits.
		ImageIn.convertTo(img, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
		ImageIn.convertTo(img_8bit, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
		//imshow("Image8bits", img);
		NomFichier = std::string("").append(imageName).append(std::string("8bits.png"));
        //imwrite(NomFichier, img);
		cv::Mat img2 = img;
		int nrows = img.rows, ncols = img.cols;
		//double threshold(InputArray src, OutputArray dst, double thresh, double maxval, int type);
		int seuil = rechercheSeuilOptimal(img, boardSize.height*boardSize.width);
		if (seuil < 0)
		{
			cout << "Aucun seuil ne convient ou pattern faux." << endl;
			return false;
		}


		cv::threshold(img, img, seuil, 255, cv::THRESH_BINARY);
		//imshow("ImageSeuillee", img);
		NomFichier = std::string("").append(imageName).append(std::string("seuillee.png"));
        //imwrite(NomFichier, img);

		//génération d'une image de synthèse pour faciliter le détection.
		cv::Mat labels, stats;
		cv::Mat centroids;//contient les centres des lampes détectées.

		connectedComponentsWithStats(img, labels, stats, centroids, 8);


		imagePoints.clear();
		imagePoints.resize(centroids.rows - 1);

		//Centroids[0] contient les coordonnées du centroid du fond (noir). On doit le retirer.
		// On commence à 1 pour enlever le fond.
		for (auto i = 1; i < centroids.rows; ++i)
		{
			imagePoints[i - 1] = cv::Point2f(centroids.at<double>(i, 0), centroids.at<double>(i, 1));
		}
		//création d'une image de synthèse pour la détection
		// On doit le faire car FindCircleGrid détecte les points dans le bon ordre et supporte la perspective.
		img = cv::Mat(ImageIn.size(), CV_8U, 255);
		for (auto i = 0; i < imagePoints.size(); ++i)
		{
			cv::circle(img, imagePoints[i], 0, cv::Scalar(0, 0, 0), -10);
		}
        NomFichier = std::string("").append(imageName).append(std::string("seuillee_point.png"));
        //imwrite(NomFichier, img);
        //2018-02-15 08-51-06.pngseuillee


		std::list<cv::Point2f> list_temp;

		for (auto i = 0; i < imagePoints.size(); ++i)
		{
			list_temp.push_back(imagePoints[i]);
		}

		cv::Mat img_interpolate = cv::Mat::zeros(nrows, ncols, CV_64FC1);
        std::vector<cv::Point2f> imagePoints_inter;
		imagePoints_inter.resize(imagePoints.size());
        /*
		int type = 1;

		if (type == 0)
		{
			gravity_center(img_8bit, img_interpolate, wind);
		}
		else if (type == 1)
		{
			bilinear_interpolation(img_8bit, img_interpolate);
		}

        get_local_maxima(wind, img_interpolate, list_temp, imagePoints_inter);*/

        std::vector<cv::Point2f> imagePoints_gc;
		imagePoints_gc.resize(imagePoints.size());
		get_local_maxima_gc(wind, img_8bit, list_temp, imagePoints_gc);

		// Set up the detector with default parameters.
		//SimpleBlobDetector detector;
		// Setup SimpleBlobDetector parameters.
		cv::SimpleBlobDetector::Params params;
		params.filterByColor = true;
		params.blobColor = 0;

        params.minRepeatability = 2;

		// Change thresholds
		params.minThreshold = 0;
		params.maxThreshold = 128;

		params.minDistBetweenBlobs = 0;
		params.thresholdStep = 1;

		// Filter by Area.
		params.filterByArea = false;

		// Filter by Circularity
		params.filterByCircularity = false;

		// Filter by Convexity
		params.filterByConvexity = false;

		// Filter by Inertia
		params.filterByInertia = false;

		// Detect blobs.
        cv::SimpleBlobDetector::create(params);


        std::vector<cv::Point2f> imagePoints_circle_grid;
		imagePoints_circle_grid.resize(imagePoints.size());


        res = findCirclesGrid(img, boardSize, imagePoints_circle_grid, cv::CALIB_CB_SYMMETRIC_GRID, cv::SimpleBlobDetector::create(params));

		//imagePoints = imagePoints_gc;

		//printf("\n Nombre de points detectes CirclesGrid = %zd\n", imagePoints.size());
		cv::Mat im_with_keypoints;

		cv::cvtColor(img2, im_with_keypoints, cv::COLOR_GRAY2BGR);
		for (auto i = 0; i < imagePoints_circle_grid.size(); ++i)
		{
			cv::circle(im_with_keypoints, imagePoints_circle_grid[i], 1, cv::Scalar(0, 0, 255));
		}

        std::vector<cv::Point2f> imagePoints_good;
		imagePoints_good.resize(imagePoints.size());

        cout << "here 7 " << endl;

		//cout << "imagePoints_circle_grid " << imagePoints_circle_grid.size() << "  imagePoints  " << imagePoints.size()  << endl;

		if (TYPE_PLAN::VERTICAL == type_plan)
		{
			for (int i = 0; i < imagePoints.size() && res; i++)
			{
				cv::Point2f val1 = imagePoints_gc[i];
				int min_j = 0;
				double the_norm;
				for (int j = 0; j < imagePoints.size(); j++)
				{
					min_j = j;
					cv::Point2f val2 = imagePoints_circle_grid[j];
					the_norm = cv::norm(val1 - val2);
					if (the_norm < 2)
					{
						break;
					}
				}
				imagePoints_good[min_j] = val1;
			}
			imagePoints = imagePoints_good;
		}
		else
		{
			imagePoints = imagePoints_gc;
		}

		return res;
	}
}
