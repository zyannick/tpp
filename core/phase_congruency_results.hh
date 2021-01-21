#pragma once

#include <Eigen/Dense>
#include <Eigen/Core>
#include <iostream>


using namespace Eigen;
using namespace std;


namespace tpp
{
    template< typename T>
    struct phase_congruency_result
    {
        phase_congruency_result()
        {

        }
        phase_congruency_result(T pcl,T pcr, T ftl,T ftr, T orl,T orr)
        {
            type_edge = 0;
            pc_first = pcl;
            pc_second = pcr;
            ft_first = ftl;
            ft_second = ftr;
            orient_first = orl;
            orient_second = orr;
        }

        phase_congruency_result(phase_congruency_result other, bool inverse)
        {
            if(inverse)
            {
                type_edge = other.type_edge;
                pc_first = other.pc_second;
                pc_second = other.pc_first;
                ft_first = other.ft_second;
                ft_second = other.ft_first;
                orient_first = other.orient_second;
                orient_second = other.orient_first;
            }
            else
            {
                type_edge = other.type_edge;
                pc_first = other.pc_first;
                pc_second = other.pc_second;
                ft_first = other.ft_first;
                ft_second = other.ft_second;
                orient_first = other.orient_first;
                orient_second = other.orient_second;
            }

        }

        phase_congruency_result(phase_congruency_result pcr1, phase_congruency_result pcr2, int side)
        {
            if(side == 1)
            {
                type_edge = pcr1.type_edge;
                pc_first = pcr1.pc_first;
                pc_second = pcr2.pc_first;
                ft_first = pcr1.ft_first;
                ft_second = pcr2.ft_first;
                orient_first = pcr1.orient_first;
                orient_second = pcr2.orient_first;
            }
            else
            {
                type_edge = pcr1.type_edge;
                pc_first = pcr1.pc_second;
                pc_second = pcr2.pc_second;
                ft_first = pcr1.ft_second;
                ft_second = pcr2.ft_second;
                orient_first = pcr1.orient_second;
                orient_second = pcr2.orient_second;
            }
        }

        void set_values(T pcl,T pcr, T ftl,T ftr, T orl,T orr)
        {
            pc_first = pcl;
            pc_second = pcr;
            ft_first = ftl;
            ft_second = ftr;
            orient_first = orl;
            orient_second = orr;
        }


        void set_values(T pcl,T pcr)
        {
            type_edge = 1;
            pc_first = pcl;
            pc_second = pcr;
        }
        int type_edge = 0;
        T pc_first, ft_first, orient_first;
        T pc_second, ft_second, orient_second;
    };





}
