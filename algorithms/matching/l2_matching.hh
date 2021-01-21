#pragma once

#include "core.hpp"
#include <Eigen/Core>
#include <list>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>


using namespace Eigen;
using namespace tpp;
 


namespace tpp
{
	inline
        float l2_matching(MatrixXf L, MatrixXf R, Vector2d pt1, Vector2d pt2, int wind_r, int wind_c)
	{
		MatrixXf l = MatrixXf::Zero(2 * wind_r + 1, 2 * wind_c + 1);
		MatrixXf r = MatrixXf::Zero(2 * wind_r + 1, 2 * wind_c + 1);

        for (int y = pt1.y() - wind_r, j = 0; y <= pt1.y() + wind_r; y++, j++)
		{
			for (int x = pt1[1] - wind_c, i = 0; x <= pt1[1] + wind_c; x++, i++)
			{
				if (x > 0 && x < L.cols() && y > 0 && y < L.rows())
					l(j, i) = L(y, x);
			}
		}

        for (int y = pt2.y() - wind_r, j = 0; y <= pt2.y() + wind_r; y++, j++)
		{
			for (int x = pt2[1] - wind_c, i = 0; x <= pt2[1] + wind_c; x++, i++)
			{
				if (x > 0 && x < R.cols() && y > 0 && y < R.rows())
					r(j, i) = R(y, x);
			}
		}
		float num = 0;
		float denum = 0;
		float diff = 0;
		for (int y = 0; y < l.rows(); y++)
		{
			for (int x = 0; x < l.cols(); x++)
			{
				diff += pow(l(y, x) - r(y, x), 2);
			}
		}


		return diff;
	}


    void matching_l2(int nrows, int ncols, MatrixXf left_mat, MatrixXf right_mat,
		std::vector<stereo_match> &list_stereo_match, cv::Mat& mat_map, float eps, int wind_row, int wind_col)
	{
        Vector3d minus_f_one;
		minus_f_one.fill(-1);
        std::vector<Vector3d> vect_sim(nrows*ncols, minus_f_one);

		//get location of maximum
		MatrixXf::Index maxRow_l, maxCol_l;
		float max_l = left_mat.maxCoeff(&maxRow_l, &maxCol_l);
		//get location of minimum
		MatrixXf::Index minRow_l, minCol_l;
		float min_l = left_mat.minCoeff(&minRow_l, &minCol_l);


		//get location of maximum
		MatrixXf::Index maxRow_r, maxCol_r;
		float max_r = right_mat.maxCoeff(&maxRow_r, &maxCol_r);
		//get location of minimum
		MatrixXf::Index minRow_r, minCol_r;
		float min_r = right_mat.minCoeff(&minRow_r, &minCol_r);

		for (int rl = 0; rl <= nrows - 0; rl++)
		{
			for (int cl = 0; cl <= ncols - 0; cl++)
			{
				float min_sim = 1000;
				if (rl >= 0 && rl < nrows && cl < ncols && cl >= 0)
				{
					mat_map.at<Vec3b>(rl, cl)[0] = uchar(left_mat(rl, cl) * 255 / max_l);
					mat_map.at<Vec3b>(rl, cl)[1] = uchar(left_mat(rl, cl) * 255 / max_l);
					mat_map.at<Vec3b>(rl, cl)[2] = uchar(left_mat(rl, cl) * 255 / max_l);
					if (left_mat(rl, cl) > eps)
					{
                        Vector2d pt_l;
						pt_l(0) = rl;
						pt_l(1) = cl;
						int idx = cl + rl* ncols;
						for (int cr = 0; cr <= ncols - 0; cr++)
						{
							if (cr >= 0 && cr < ncols)
							{
								mat_map.at<Vec3b>(rl + nrows, cr + ncols)[0] = uchar(right_mat(rl, cr) * 255 / max_r);
								mat_map.at<Vec3b>(rl + nrows, cr + ncols)[1] = uchar(right_mat(rl, cr) * 255 / max_r);
								mat_map.at<Vec3b>(rl + nrows, cr + ncols)[2] = uchar(right_mat(rl, cr) * 255 / max_r);
								if (right_mat(rl, cr) > eps)
								{
                                    Vector2d pt_r;
									pt_r(0) = rl;
									pt_r(1) = cr;
									//float s = similarity_matching(left_mat, left_mat, pt_l, pt_r, wind_row, wind_col);
									float s = l2_matching(left_mat, right_mat, pt_l, pt_r, wind_row, wind_col);
									if (s < min_sim)
									{
										//sim(rl, cl) = pt_r;
                                        vect_sim[idx].y() = pt_r.y();
                                        vect_sim[idx].x() = pt_r.x();
                                        vect_sim[idx].z() = min_sim;
										min_sim = s;
									}
								}
							}
						}
                        Vector2d m_r;
                        m_r << vect_sim[idx].x(), vect_sim[idx].y();
						list_stereo_match.push_back(stereo_match(pt_l, m_r, vect_sim[idx][2]));
					}
				}
			}
		}
        std::sort(list_stereo_match.begin(), list_stereo_match.end(), crescendo_similarity());
	}
}
