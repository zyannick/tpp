#pragma once


#define _USE_MATH_DEFINES

#define _USE_VPP


#include <algorithm>
#include "opencv2/highgui.hpp"
#include "algorithms/miscellaneous.hh"

#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <fftw3.h>

#include "gabor_util.hh"
#include "phase_congruency_utils.hh"
#include "core.hpp"

#define  FFTW_IN_PLACE
#pragma comment(lib, "fftw3f")

#include <time.h>
#include <chrono>

using namespace std;
 
using namespace Eigen;
using namespace std::chrono;

#ifdef vpp
#include "vpp/vpp.hh"
using namespace vpp;
#endif

namespace tpp {




#ifdef vpp
template <typename source,typename target>
void phase_congruency_opt(image2d<source> img,
                          phase_congruency_output_vpp<target> &pc_out)
{
    int nrows = img.nrows(), ncols = img.ncols();

    /// convolution by FFT in the freq-domain
    /// in-place transformation, both before and after FFT
    fftwf_complex *m_imgfftConv;
    fftwf_complex *m_imgfftConv_;

    /// the FFTW plan for inverse FFT
    fftwf_plan m_invplan;

    // image FFT variables
    float *image_in;
    fftwf_complex *m_fftwOut;
    fftwf_complex *m_imgfft;
    fftwf_plan m_fwplan;

    // variables for image FFT
    // the most important is the FFTW output memory layout
    int nw2 = ncols / 2 + 1;
    int fftwSz = nrows * nw2;
    m_fftwOut = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * fftwSz);
    m_imgfft = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * (nrows*ncols));
    m_imgfftConv = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * (nrows*ncols));
    m_imgfftConv_ = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * (nrows*ncols));

    image_in = (target *)malloc (sizeof(target) * (nrows*ncols));
    image2d_to_array<source, target>(img, image_in);
    //display_image2d(img);

    // special note: the input should be row major
    m_fwplan = fftwf_plan_dft_r2c_2d(nrows, ncols, image_in, m_fftwOut, FFTW_ESTIMATE);
    fftwf_execute(m_fwplan);
    hermite_to_array_pc(m_fftwOut, m_imgfft, ncols, nrows);

    m_invplan = fftwf_plan_dft_2d(nrows, ncols, m_imgfftConv, m_imgfftConv, FFTW_BACKWARD, FFTW_ESTIMATE);

    image2d<std::complex<target>> image_out_fft(nrows, ncols);
    pixel_wise(image_out_fft) | [&] (auto& i) { i = std::complex<target>(0,0); };
    complex_array_image2d<float>(image_out_fft, m_imgfft);

    image2d<target> covx2 = initialise_image2d(nrows, ncols, target(0));
    image2d<target> covy2 = initialise_image2d(nrows, ncols, target(0));
    image2d<target> covxy = initialise_image2d(nrows, ncols, target(0));

    std::vector<image2d<target>> EnergyV(3);
    EnergyV[0] = initialise_image2d(nrows, ncols, target(0));
    EnergyV[1] = initialise_image2d(nrows, ncols, target(0));
    EnergyV[2] = initialise_image2d(nrows, ncols, target(0));


    for (int orient = 0; orient < pc_out.norient; orient++) // rows
    {

        target angl = orient * M_PI / pc_out.norient; //Filter angle.
        image2d<target> spread = pc_out.list_spread[orient];

        //Initialize accumulator matrices.

        image2d<target> sumE_ThisOrient = initialise_image2d(nrows, ncols, target(0));        // Initialize accumulator matrices.
        image2d<target> sumO_ThisOrient = initialise_image2d(nrows, ncols, target(0));
        image2d<target> sumAn_ThisOrient = initialise_image2d(nrows, ncols, target(0));
        image2d<target> maxAn = initialise_image2d(nrows, ncols, target(0));
        image2d<target> Energy = initialise_image2d(nrows, ncols, target(0));


        float tau = 0;



        //For each scale...
        for (int scale = 0; scale < pc_out.nscale; scale++)
        {

            image2d<std::complex<target>> fft_img_convolved(nrows, ncols);
            pixel_wise(fft_img_convolved, spread, pc_out.logGabor[scale], image_out_fft) | [&]
                    (auto &out, auto sp, auto loga, auto ift)
            {
                out = ift * (sp * loga);
            };

            image2d_to_complex_array(fft_img_convolved, m_imgfftConv);
            fftwf_execute(m_invplan);

            int offset = orient * pc_out.nscale + scale;
            complex_array_image2d(pc_out.EO[offset], m_imgfftConv);

            pc_out.EO[offset] = pc_out.EO[offset] / (target(nrows *ncols));



            image2d<target> An = image2d<target>(nrows, ncols);
            pixel_wise(An, pc_out.EO[offset]) | [&]
                    (auto &a, auto eo)
            {
                a = target(sqrt(eo.real()* eo.real() + eo.imag()* eo.imag()));
            };


            pixel_wise(sumAn_ThisOrient, sumE_ThisOrient, sumO_ThisOrient, An, pc_out.EO[offset]) | [&]
                    (auto &sta, auto &sum_e, auto &sum_o, auto an, auto eo)
            {
                sta = sta + an;
                sum_e = sum_e + eo.real();
                sum_o = sum_o + eo.imag();
            };


            if (scale == 0)
            {
                if (pc_out.pc_settings.noiseMethod == -1) //Use median to estimate noise statistics
                {
                    VectorXf temp_vec(sumAn_ThisOrient.ncols()*sumAn_ThisOrient.nrows());
                    pixel_wise(sumAn_ThisOrient, sumAn_ThisOrient.domain()) | [&]
                            (auto sta, auto coord)
                    {
                        int row = coord[0];
                        int col = coord[1];
                        temp_vec(row + col * nrows) = sta;
                    };
                    tau = median_vector(temp_vec) / sqrt_log_4;
                }
                else if (pc_out.pc_settings.noiseMethod == -2) // Use mode to estimate noise statistics
                {
                    VectorXf temp_vec(sumAn_ThisOrient.ncols()*sumAn_ThisOrient.nrows());
                    pixel_wise(sumAn_ThisOrient, sumAn_ThisOrient.domain()) | [&]
                            (auto sta, auto coord)
                    {
                        int row = coord[0];
                        int col = coord[1];
                        temp_vec(row + col * nrows) = sta;
                    };
                    tau = rayleigh_mode(temp_vec);
                }

                maxAn = An;
            }
            else
            {
                pixel_wise(maxAn, An) | [&] (auto &ma, auto an)
                {
                    ma = max(ma, an);
                };
            }
        }

        //cout << "maxAn " << endl;
        //display_image2d(maxAn);

        EnergyV[0] = EnergyV[0] + sumE_ThisOrient;
        EnergyV[1] = EnergyV[1] + cos(angl)*sumO_ThisOrient;
        EnergyV[2] = EnergyV[2] + sin(angl)*sumO_ThisOrient;

        //cout << "EnergyV[0] " << endl;
        //display_image2d(EnergyV[0]) ;

        //cout << "EnergyV[1] " << endl;
        //display_image2d(EnergyV[1]);

        //cout << "EnergyV[2] " << endl;
        //display_image2d(EnergyV[2]);



        image2d<target> XEnergy = image2d<target>(nrows, ncols);
        image2d<target> MeanE = image2d<target>(nrows, ncols);
        image2d<target> MeanO = image2d<target>(nrows, ncols);
        pixel_wise(XEnergy, MeanE, MeanO, sumE_ThisOrient, sumO_ThisOrient) | [&]
                (auto &ener, auto &me, auto &mo, auto sum_e, auto sum_o)
        {
            ener = target(sqrt(sum_e*sum_e + sum_o*sum_o) + pc_out.pc_settings.epsilon);
            me = target(sum_e / ener);
            mo = target(sum_o / ener);
        };


        for (int scale = 0; scale < pc_out.nscale; scale++)
        {
            int offset = orient * pc_out.nscale + scale;

            pixel_wise(Energy, pc_out.EO[offset], MeanE, MeanO) | [&] (auto &ener, auto eo, auto me, auto mo)
            {
                auto e = eo.real();
                auto o = eo.imag();
                ener = ener + e*me + o*mo - target(fabs(e*mo - o*me ));
            };
            //cout << "Energy " << endl;
            //display_image2d(Energy);
        }

        float T = 0;
        if (pc_out.pc_settings.noiseMethod >= 0) //We are using a fixed noise threshold
        {
            T = pc_out.pc_settings.noiseMethod; //use supplied noiseMethod value as the threshold

        }
        else
        {
            float totalTau = tau * (1 - pow(1 / pc_out.pc_settings.mult, pc_out.pc_settings.nscale)) / (1 - (1 / pc_out.pc_settings.mult));
            float EstNoiseEnergyMean = totalTau*sqrt(M_PI / 2);        // Expected mean and std
            float EstNoiseEnergySigma = totalTau*sqrt((4 - M_PI) / 2);   // values of noise energy
            T = EstNoiseEnergyMean + pc_out.pc_settings.k*EstNoiseEnergySigma; // Noise threshold
        }

        image2d<target> wdth = image2d<target>(nrows, ncols);
        image2d<target> weight = image2d<target>(nrows, ncols);


        pixel_wise(Energy, wdth, weight, sumAn_ThisOrient, maxAn) | [&]
                (auto &ener, auto &wd, auto &wth, auto sao, auto ma)
        {
            ener = max((ener - target(T)), target(0));
            wd = (sao / (ma + pc_out.pc_settings.epsilon) - 1 / (pc_out.pc_settings.nscale - 1));
            wth = 1/( 1 + exp(pc_out.pc_settings.g * (pc_out.pc_settings.cutt_off - wd)) );
        };



        pixel_wise(pc_out.PC[orient], pc_out.pcSum, Energy, weight, sumAn_ThisOrient) | [&]
                (auto &pc, auto &pcs, auto ener, auto wt, auto sao)
        {
            pc = (wt * ener) / sao;
            pcs = pcs + pc;
        };

        pixel_wise(covx2, covy2, covxy, pc_out.PC[orient]) | [&]
                (auto &x2, auto &y2, auto &xy, auto pc)
        {
            auto covx = pc * cos(angl);
            auto covy = pc * sin(angl);
            x2 = x2 + pow(covx, 2);
            y2 = y2 + pow(covy, 2);
            xy = xy + covx * covy;
        };/**/

    }


    pixel_wise(covx2, covy2, covxy, pc_out.m, pc_out.M, pc_out.featType, EnergyV[0], EnergyV[1], EnergyV[2]) | [&]
            (auto &x2, auto &y2, auto &xy, auto &m, auto &M, auto &ft, auto ener0, auto ener1, auto ener2)
    {
        x2 = x2 / (pc_out.pc_settings.norient / 2);
        y2 = y2 / (pc_out.pc_settings.norient / 2);
        xy = xy * target( 4 / pc_out.pc_settings.norient);
        target denom = sqrt(pow(xy, 2) + pow(x2-y2, 2)) + pc_out.pc_settings.epsilon;
        M = (y2 + x2 + denom) / 2;
        m = (y2 + x2 - denom) / 2;

        target oddV = sqrt(ener1*ener1 + ener2*ener2);
        ft = atan2(ener0, oddV) * (180 / M_PI);
    };


    //display_image2d(pc_out.M);
    //display_image2d(pc_out.m);

}
#endif




}
