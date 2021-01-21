#pragma once


#include <Eigen/Core>
#include <list>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include "utils_matching.hh"
#include "core.hpp"
#include <chrono>

using namespace Eigen;
using namespace tpp;
 
using namespace std::chrono;

#ifdef vpp
#include <vpp/vpp.hh>
#include <vpp/utils/opencv_bridge.hh>
#include <vpp/utils/opencv_utils.hh>
using namespace vpp;
#endif

namespace tpp
{




inline
float compute_lades_similarity_lines(MatrixXf l, MatrixXf r)
{
    float num = 0;
    float denum_r = 0;
    float denum_l = 0;

    for (int y = 0; y < l.rows(); y++)
    {
        for (int x = 0; x < l.cols(); x++)
        {
            num += l(y, x) * r(y, x);
            denum_r += r(y, x) * r(y, x);
            denum_l += l(y, x) *  l(y, x);
        }
    }

    float denum = denum_r * denum_l;
    return num / sqrt(denum);
}


#ifdef vpp
inline
float compute_lades_similarity_lines(image2d<float> l, image2d<float> r)
{
    float num = 0;
    float denum_r = 0;
    float denum_l = 0;
    for (int y = 0; y < l.nrows(); y++)
    {
        for (int x = 0; x < l.ncols(); x++)
        {
            //cout << "voila " << l.nrows()  << "   " <<  l.ncols()  << "  "<< l(y, x)  << "   "  << r(y, x)  << endl;
            num += l(y, x) * r(y, x);
            denum_r += r(y, x) * r(y, x);
            denum_l += l(y, x) *  l(y, x);

        }
    }

    //cout << "compute_lades_similarity_lines " << endl;

    //float denum = denum_r * denum_l +1;
    //return num / sqrt(denum);
    //cout << num << "   " <<  denum_l << "  "  << denum_r << endl;
    return num / sqrt(denum_r * denum_l);
}
#endif








typedef std::tuple<int, int, int> key_type;



template< typename Type>
void matching_similarity( std::vector<stereo_match> &list_stereo_match,
                          phase_congruency_result<MatrixXf> pcr,
                          base_options<Type> stm, int side, std::map<key_type, float> &map_sim,
                          MatrixXf otsu_first = MatrixXf::Zero(0,0),
                          MatrixXf otsu_second = MatrixXf::Zero(0,0))
{

    MatrixXf img_left ,img_right;
    int pad = 2*(stm.wind_row+stm.wind_col);
    fill_border_zero(pcr.pc_first, img_left, pad);
    fill_border_zero(pcr.pc_second, img_right, pad);
    long nrows = pcr.pc_first.rows();
    long ncols = pcr.pc_first.cols();


    Vector3d minus_f_one;
    minus_f_one.fill(-1);

    std::vector<Vector3d> best_matches(2);

    for(int i = 0; i < best_matches.size() ; i++)
    {
        best_matches[i].fill(-1);
    }

    int nb_features = 0;

    int disparity_range_mid = 5;

    //#pragma omp parallel for
    for (int sr = 0; sr < nrows; sr++)
    {
        for (int sl = 0; sl < ncols ; sl++)
        {
            if (pcr.pc_first(sr, sl) > stm.eps)
                if ((otsu_first.rows() > 0 && otsu_first(sr, sl)>0) || otsu_first.rows() == 0)
                {

                    float max_sim = -1;
                    Vector2d first;
                    first.y() = sr;
                    first.x() = sl;

                    double second_row;
                    Vector3d temp;
                    temp.fill(-1);


                    for(int second_col = first.x() - disparity_range_mid  ; second_col < first.x() + disparity_range_mid ; second_col++)
                    {
                        //cout << "col " << second_col << "   before " << first.x() <<  endl;
                        if(second_col > 0 && second_col < ncols)
                        {
                            second_row = first.y();
                            int slice = second_row;
                            for(slice = second_row - 3 ; slice <= second_row + 3 ; slice++)
                            {
                                if(slice >=0 && slice < nrows)
                                {
                                    if (pcr.pc_second(slice, second_col) > stm.eps)
                                        if ((otsu_second.rows() > 0 && otsu_second(sr, sl)>0) || otsu_second.rows() == 0)
                                        {
                                            Vector2d second;
                                            second.y() = slice;
                                            second.x() = second_col;


                                            //Center of the matrix
                                            Vector2i center_first(first.y()+pad, first.x()+pad);
                                            //Get the debut of the matrix
                                            Vector2i start_first(center_first.y()-stm.wind_row, center_first.x()-stm.wind_row);
                                            //out << "Padding " << pad << "  before "  << first.transpose()  << " after " << start_first.transpose() << endl;
                                            MatrixXf first_block = img_left.block (start_first.y(), start_first.x(), 2*stm.wind_row+1, 2*stm.wind_col+1);

                                            //Center of the matrix
                                            Vector2i center_second(second.y()+pad, second.x()+pad);
                                            //Get the debut of the matrix
                                            Vector2i start_second(center_second.y()-stm.wind_row, center_second.x()-stm.wind_row);
                                            MatrixXf second_block = img_right.block(start_second.y(), start_second.x(), 2*stm.wind_row+1, 2*stm.wind_col+1);

                                            //cout << "taille " << first_block.rows() << "  " << first_block.cols() << " ------- " << second_block.rows() << "  " << second_block.cols()  << endl;
                                            int k1 = 0;
                                            int k2 = 0;
                                            if(side == 1)
                                            {
                                                k1 = 1;
                                                k2 = 2;
                                            }
                                            else
                                            {
                                                k1 = 2;
                                                k2 = 1;
                                            }

                                            auto search_1 = map_sim.find(std::make_tuple(2, first.y(), first.x() ));
                                            auto search_2 = map_sim.find(std::make_tuple(1, second.y(), second.x() ));

                                            float denum1;
                                            float denum2;

                                            if(search_1 != map_sim.end())
                                            {
                                                denum1 = search_1->second;
                                            }
                                            else
                                            {
                                                denum1 = (first_block.binaryExpr(first_block, mymulpli<float>())).sum();
                                            }


                                            float num = (first_block.binaryExpr(second_block, mymulpli<float>())).sum();

                                            denum2 = (second_block.binaryExpr(second_block, mymulpli<float>())).sum();
                                            float denum = denum1 * denum2;
                                            float s = num / denum;
                                            //float s_ = compute_lades_similarity_lines(first_block, second_block);

                                            if (s > max_sim)
                                            {
                                                temp.y() = second.y();
                                                temp.x() = second.x();
                                                temp[2] = s;
                                                max_sim = s;
                                            }

                                            if(best_matches[1](2) < s)
                                            {
                                                if(best_matches[0](2) < s)
                                                {
                                                    best_matches[0].y() = second.y();
                                                    best_matches[0].x() = second.x();
                                                    best_matches[0](2) = s;
                                                }
                                                else
                                                {
                                                    best_matches[0].y() = second.y();
                                                    best_matches[0].x() = second.x();
                                                    best_matches[1](2) = s;
                                                }
                                            }
                                        }
                                }
                            }
                        }

                    }


                    if(max_sim == -1)
                        continue;

                    Vector2d m_r;
                    m_r.x() = temp.x();
                    m_r.y() = temp.y();

                    if(max_sim > 0.1)
                    {
                        float strength = ( pcr.pc_first(first.y(),first.x()) + pcr.pc_second( m_r.y() , m_r.x() ) ) /2;

                        //cout << "First match " <<  best_matches[0].segment(0,2).transpose() << "    second  match  " << best_matches[1].segment(0,2).transpose()  << endl;
                        Vector2d v1 = best_matches[0].segment(0,2);
                        Vector2d v2 = best_matches[1].segment(0,2);
                        float dist1 = (first - v1).norm();
                        float dist2 = (first - v2).norm();

                        float lowe_ratio = dist1 / dist2;
                        //cout << "La similarite "  << temp[2] << endl;
                        if(side==1)
                        {
                            //cout << "left to right " << first << "  "  << m_r << endl << endl;
                            list_stereo_match.push_back(stereo_match(first, m_r, temp.z(), strength , lowe_ratio));
                        }
                        else if(side==2)
                        {
                            //cout << "right to left " << first << "  "  << first << endl << endl;
                            list_stereo_match.push_back(stereo_match(m_r, first, temp.z(), strength, lowe_ratio ));
                        }
                    }


                    nb_features++;
                }
        }
    }

    //cout << "nb_features " << nb_features << endl;

}


}
