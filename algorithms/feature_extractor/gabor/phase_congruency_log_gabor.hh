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

#define  FFTW_IN_PLACE
#pragma comment(lib, "fftw3f")

#include <time.h>
#include <chrono>

using namespace std;
 
using namespace Eigen;
using namespace std::chrono;

namespace tpp {


int here = 0;

void display_here()
{
    cout << "here " << here++ << endl;
}

template< typename Type>
void phase_congruency_3(MatrixXf img, phase_congruency_output_eigen<Type> &pco)
{

    auto nrows = img.rows();
    auto ncols = img.cols();

    //cout << "nrows " << nrows << "   ncols " << ncols << endl;



    double sqrt_log_4 = sqrt(log(4));



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
    auto nw2 = ncols / 2 + 1;
    auto fftwSz = nrows * nw2;
    m_fftwOut = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * fftwSz);
    m_imgfft = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * (nrows*ncols));
    m_imgfftConv = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * (nrows*ncols));
    m_imgfftConv_ = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * (nrows*ncols));



    image_in = (float *)malloc (sizeof(float) * (nrows*ncols));
    eigen_matrix_to_float_array(img, image_in);

    // special note: the input should be row major
    m_fwplan = fftwf_plan_dft_r2c_2d(nrows, ncols, image_in, m_fftwOut, FFTW_ESTIMATE);
    fftwf_execute(m_fwplan);
    hermite_to_array_pc(m_fftwOut, m_imgfft, ncols, nrows);

    m_invplan = fftwf_plan_dft_2d(nrows, ncols, m_imgfftConv, m_imgfftConv, FFTW_BACKWARD, FFTW_ESTIMATE);

    MatrixXcf image_out_fft = MatrixXcf::Zero(nrows, ncols);

    complex_array_eigen_matrix(image_out_fft, m_imgfft);


    MatrixXf covx2 = MatrixXf::Zero(nrows, ncols);
    MatrixXf covy2 = MatrixXf::Zero(nrows, ncols);
    MatrixXf covxy = MatrixXf::Zero(nrows, ncols);

    std::vector<MatrixXf> EnergyV(3);
    EnergyV[0] = MatrixXf::Zero(nrows, ncols);
    EnergyV[1] = MatrixXf::Zero(nrows, ncols);
    EnergyV[2] = MatrixXf::Zero(nrows, ncols);

    for (int orient = 0; orient < pco.pc_settings.norient; orient++) // rows
    {

        //Construct the angular filter spread function
        float angl = orient * M_PI / pco.pc_settings.norient; //Filter angle.

        auto spread = pco.list_spread[orient];



        MatrixXf sumE_ThisOrient = MatrixXf::Zero(nrows, ncols);          // Initialize accumulator matrices.
        MatrixXf sumO_ThisOrient = MatrixXf::Zero(nrows, ncols);
        MatrixXf sumAn_ThisOrient = MatrixXf::Zero(nrows, ncols);
        MatrixXf maxAn = MatrixXf::Zero(nrows, ncols);
        MatrixXf Energy = MatrixXf::Zero(nrows, ncols);



        float tau = 0;

        //For each scale...
        for (int scale = 0; scale < pco.pc_settings.nscale; scale++)
        {

            MatrixXcf fft_img_convolved = image_out_fft.binaryExpr(
                        (pco.logGabor[scale]).binaryExpr(spread, mymulpli<Type>())
                        , mymulpli_complex_float<std::complex<Type>>());
            //cout << fft_img_convolved << endl;
            eigen_matrix_to_complex_array(fft_img_convolved, m_imgfftConv);
            fftwf_execute(m_invplan);



            int offset = orient * pco.pc_settings.nscale + scale;

            complex_array_eigen_matrix(pco.EO[offset], m_imgfftConv);
            pco.EO[offset] = pco.EO[offset] / (float(nrows *ncols));

            //cout << "EO[offset] " << endl << pco.EO[offset] << endl;

            MatrixXf An = (((pco.EO[offset]).real()).cwiseAbs2() + ((pco.EO[offset]).imag()).cwiseAbs2()).cwiseSqrt();
            //cout << "An " << endl << An << endl;
            sumAn_ThisOrient = sumAn_ThisOrient + An;
            sumE_ThisOrient = sumE_ThisOrient + (pco.EO[offset]).real();
            sumO_ThisOrient = sumO_ThisOrient + (pco.EO[offset]).imag();


            if (scale == 0)
            {
                if (pco.pc_settings.noiseMethod == -1) //Use median to estimate noise statistics
                {
                    VectorXf temp_vec(Map<VectorXf>(sumAn_ThisOrient.data(), sumAn_ThisOrient.cols()*sumAn_ThisOrient.rows()));
                    tau = median_vector(temp_vec) / sqrt_log_4;

                }
                else if (pco.pc_settings.noiseMethod == -2) // Use mode to estimate noise statistics
                {
                    VectorXf temp_vec(Map<VectorXf>(sumAn_ThisOrient.data(), sumAn_ThisOrient.cols()*sumAn_ThisOrient.rows()));
                    tau = rayleigh_mode(temp_vec);

                }

                maxAn = An;
            }
            else
            {

                maxAn = maxAn.cwiseMax(An);
            }

        }

        //cout << "maxAn " << endl <<  maxAn << endl << endl;

        EnergyV[0] = EnergyV[0] + sumE_ThisOrient;
        EnergyV[1] = EnergyV[1] + cos(angl)*sumO_ThisOrient;
        EnergyV[2] = EnergyV[2] + sin(angl)*sumO_ThisOrient;

        //cout << "EnergyV[0] " << endl <<  EnergyV[0] << endl << endl;

        //cout << "EnergyV[1] " << endl <<  EnergyV[1] << endl << endl;

        //cout << "EnergyV[2] " << endl <<  EnergyV[2] << endl << endl;



        MatrixXf XEnergy = (sumE_ThisOrient.cwiseAbs2() + sumO_ThisOrient.cwiseAbs2()).cwiseSqrt() + pco.matrix_epsilon;
        MatrixXf MeanE = sumE_ThisOrient.binaryExpr(XEnergy, mydivide<float>());
        MatrixXf MeanO = sumO_ThisOrient.binaryExpr(XEnergy, mydivide<float>());

        for (int scale = 0; scale < pco.pc_settings.nscale; scale++)
        {
            int offset = orient * pco.pc_settings.nscale + scale;
            MatrixXf E = (pco.EO[offset]).real();
            MatrixXf O = (pco.EO[offset]).imag();
            Energy = Energy + E.binaryExpr(MeanE, mymulpli<float>()) + O.binaryExpr(MeanO, mymulpli<float>())
                - (E.binaryExpr(MeanO, mymulpli<float>()) - O.binaryExpr(MeanE, mymulpli<float>())).cwiseAbs();
            //cout << "Energy " << endl <<  Energy << endl << endl;
        }

        float T = 0;
        if (pco.pc_settings.noiseMethod >= 0) //We are using a fixed noise threshold
        {
            T = pco.pc_settings.noiseMethod; //use 82252supplied noiseMethod value as the threshold

        }
        else
        {

            float totalTau = tau * (1 - pow(1 / pco.pc_settings.mult, pco.pc_settings.nscale)) / (1 - (1 / pco.pc_settings.mult));
            float EstNoiseEnergyMean = totalTau*sqrt(M_PI / 2);        // Expected mean and std
            float EstNoiseEnergySigma = totalTau*sqrt((4 - M_PI) / 2);   // values of noise energy
            T = EstNoiseEnergyMean + pco.pc_settings.k*EstNoiseEnergySigma; // Noise threshold
        }

        Energy = (Energy - MatrixXf::Constant(nrows, ncols, T)).cwiseMax(MatrixXf::Zero(nrows, ncols));


        MatrixXf wdth = (sumAn_ThisOrient.binaryExpr(maxAn + pco.matrix_epsilon, mydivide<float>()) - MatrixXf::Ones(nrows, ncols)) / (pco.pc_settings.nscale - 1);

        MatrixXf mat_cutoff = MatrixXf::Constant(nrows, ncols, pco.pc_settings.cutt_off);
        MatrixXf mat_temp_ = MatrixXf::Ones(nrows, ncols) + (pco.pc_settings.g*(mat_cutoff - wdth)).unaryExpr(myexpo<float>());
        MatrixXf weight = (MatrixXf::Ones(nrows, ncols)).binaryExpr(mat_temp_, mydivide<float>());


        pco.PC[orient] = (weight.binaryExpr(Energy, mymulpli<float>())).binaryExpr(sumAn_ThisOrient, mydivide<float>());
        //cout << pco.PC[orient] << endl;
        pco.pcSum = pco.pcSum + pco.PC[orient];

        MatrixXf covx = pco.PC[orient] * cos(angl);
        MatrixXf covy = pco.PC[orient] * sin(angl);
        covx2 = covx2 + covx.cwiseAbs2();
        covy2 = covy2 + covy.cwiseAbs2();
        covxy = covxy + covx.binaryExpr(covy, mymulpli<float>());


    }

    covx2 = covx2 / (pco.pc_settings.norient / 2);
    covy2 = covy2 / (pco.pc_settings.norient / 2);
    covxy = 4 * covxy / pco.pc_settings.norient; // This gives us 2*covxy/(norient/2)



    MatrixXf denom = (covxy.cwiseAbs2() + (covx2 - covy2).cwiseAbs2()).cwiseSqrt() + pco.matrix_epsilon;
    pco.M = (covy2 + covx2 + denom) / 2;          // Maximum moment
    pco.m = (covy2 + covx2 - denom) / 2;          // ... and minimum moment




    MatrixXf OddV = (EnergyV[1].cwiseAbs2() + EnergyV[2].cwiseAbs2()).cwiseSqrt();
    pco.featType = (EnergyV[0]).binaryExpr(OddV, my_atan2<float>());
    pco.featType = pco.featType * (180 / M_PI);

}


}
