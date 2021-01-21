#pragma once


#define _USE_MATH_DEFINES


#include <algorithm>
#include "opencv2/highgui.hpp"
#include "algorithms/miscellaneous.hh"
#include "algorithms/morphology.hh"

#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include "fftw3.h"
#include "gabor_util.hh"

#define  FFTW_IN_PLACE
#pragma comment(lib, "fftw3f")

using namespace std;
 
using namespace Eigen;

namespace tpp {

float epsilon = 0.0001;


struct phase_congruency_settings
{
    phase_congruency_settings(int nr,int nc): nrows(nr),ncols(nc) {
        is_initialized = true;
        matrix_two = MatrixXf::Constant(nrows, ncols, 2); //op
        matrix_pi = MatrixXf::Constant(nrows, ncols, M_PI); //op
        matrix_epsilon = MatrixXf::Constant(nrows, ncols, epsilon); //op
        lp =  low_pass_filter(nrows, ncols, 0.45, 15);
    }
    int nrows;
    int ncols;
    bool is_initialized = false;
    MatrixXf matrix_two;
    MatrixXf matrix_pi;
    MatrixXf matrix_epsilon;
    MatrixXf lp;

};


void perfft2(MatrixXcf &P, MatrixXcf S, MatrixXf &p, MatrixXf &s, MatrixXf img)
{
    s = MatrixXf::Zero(img.rows(), img.cols());
    int nrows = img.rows(), ncols = img.cols();
    MatrixXf matrix_two = MatrixXf::Constant(nrows, ncols, 2);


    //Compute the boundary image which is equal to the image discontinuity
    //values across the boundaries at the edges and is 0 elsewhere
    s.row(0) = img.row(0) - img.row(nrows - 1);
    s.row(nrows - 1) = -s.row(0);
    s.col(0) = s.col(0) + img.col(0) - img.col(ncols - 1);
    s.col(ncols - 1) = s.col(ncols - 1) - img.col(0) + img.col(ncols - 1);

    float *s_p = (float *)malloc (sizeof(float) * (nrows*ncols));
    float *img_p = (float *)malloc (sizeof(float) * (nrows*ncols));

    eigen_matrix_to_float_array(s, s_p);
    eigen_matrix_to_float_array(img, img_p);

    fftwf_complex *m_fftw_s_out;
    fftwf_complex *m_fftw_s_out_mat;
    fftwf_plan m_fwplan_s;

    int nw2 = ncols / 2 + 1;
    int fftwSz = nrows * nw2;
    m_fftw_s_out = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * fftwSz);
    m_fftw_s_out_mat = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * (nrows*ncols));

    //Generate grid upon which to compute the filter for the boundary image in
    //the frequency domain.Note that cos() is cyclic hence the grid values can
    // range from 0 .. 2*pi rather than 0 .. pi and then pi .. 0
    MatrixXf cx, cy;
    VectorXf temp1(ncols), temp2(nrows);
    for (int i = 0; i < ncols; i++)
        temp1[i] = 2 * M_PI*i;
    for (int i = 0; i < nrows; i++)
        temp2[i] = 2 * M_PI*i;

    temp1 = temp1 / (float)ncols;
    temp2 = temp2 / (float)nrows;

    mesgrid(temp1, temp2, cx, cy);

    //cout << "cx " << endl;
    //cout << cx << endl;

    //cout << "cy " << endl;
    //cout << cy << endl;

    //Generate FFT of smooth component
    m_fwplan_s = fftwf_plan_dft_r2c_2d(nrows, ncols, s_p, m_fftw_s_out, FFTW_ESTIMATE);
    fftwf_execute(m_fwplan_s);
    MatrixXcf Eig_m_fftw_s_out = MatrixXcf::Zero(nrows, ncols);
    hermite_to_array_pc(m_fftw_s_out, m_fftw_s_out_mat, ncols, nrows);
    complex_array_eigen_matrix(Eig_m_fftw_s_out, m_fftw_s_out_mat);
    MatrixXf temp_m = 2 * (matrix_two - cx.unaryExpr(mycosine<float>()) - cy.unaryExpr(mycosine<float>()));
    S = Eig_m_fftw_s_out.binaryExpr(temp_m, mydivid_complex_float<std::complex<float>>());
    S(0, 0) = 0;

    //cout << "S " << endl;
    //cout << S << endl;

    fftwf_complex *m_fftw_img_out;
    fftwf_complex *m_fftw_img_out_mat;
    fftwf_plan m_fwplan_p;
    m_fftw_img_out = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * fftwSz);
    m_fftw_img_out_mat = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * (nrows*ncols));
    m_fwplan_p = fftwf_plan_dft_r2c_2d(nrows, ncols, img_p, m_fftw_img_out, FFTW_ESTIMATE);
    fftwf_execute(m_fwplan_p);
    hermite_to_array_pc(m_fftw_img_out, m_fftw_img_out_mat, ncols, nrows);
    Eig_m_fftw_s_out = MatrixXcf::Zero(nrows, ncols);
    complex_array_eigen_matrix(Eig_m_fftw_s_out, m_fftw_img_out_mat);
    P = Eig_m_fftw_s_out - S;

    //cout << "P " << endl;
    //cout << P << endl;

    /*fftwf_plan m_invplan;
    fftwf_complex *m_imgfftConv;
    m_imgfftConv = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * (nrows*ncols));
    m_invplan = fftwf_plan_dft_2d(nrows, ncols, m_imgfftConv, m_imgfftConv, FFTW_BACKWARD, FFTW_ESTIMATE);
    eigen_matrix_to_complex_array(S, m_imgfftConv);
    fftwf_execute(m_invplan);
    MatrixXcf ifft_S = MatrixXcf::Zero(nrows, ncols);
    complex_array_eigen_matrix(ifft_S, m_imgfftConv);
    s = ifft_S.real();
    p = img - s;

    cout << "s " << endl;
    cout << s << endl;

    cout << "p " << endl;
    cout << p << endl;*/
}



void phasecongmono_optimized(MatrixXf img, int nscale, int norient, float min_wave_length,
    float mult, float sigma__onf, float k, float cutt_off, int noiseMethod, float g,
    float deviation_gain, MatrixXf &PC, MatrixXf &ft, float &T, MatrixXf &orient,phase_congruency_settings pcs)
{

    int nrows = img.rows(), ncols = img.cols();
    double sqrt_log_4 = sqrt(log(4));
    MatrixXcf IM, S;
    MatrixXf p, s;
    perfft2(IM, S, p, s, img);

    MatrixXf sumAn = MatrixXf::Zero(nrows, ncols);
    MatrixXf sumf = MatrixXf::Zero(nrows, ncols);
    MatrixXf sumh1 = MatrixXf::Zero(nrows, ncols);
    MatrixXf sumh2 = MatrixXf::Zero(nrows, ncols);
    MatrixXf radius, u1, u2;
    filter_grid(nrows, ncols, radius, u1, u2);
    radius(0, 0) = 1;
    const std::complex<float> If(0.0f, 1.0f);
    MatrixXcf u1u2 = MatrixXcf::Zero(nrows, ncols);
    u1u2 = -u2 + u1*If;
    //u1u2 = u1u2.binaryExpr(-u2, set_real_part());
    //u1u2 = u1u2.binaryExpr(u1, set_imag_part());
    MatrixXcf H = u1u2.binaryExpr(radius, mydivid_complex_float<std::complex<float>>());
    //cout << "H " << endl;
    //cout << H << endl;
    MatrixXf lp = pcs.lp;;  //op
    MatrixXf logGabor;
    MatrixXf maxAn;
    MatrixXf matrix_two = pcs.matrix_two; //op
    MatrixXf matrix_pi = pcs.matrix_pi; //op
    MatrixXf matrix_epsilon = pcs.matrix_epsilon; //op
    float tau;
    for (int scale = 0; scale < nscale; scale++)
    {
        float wave_length = min_wave_length * pow(mult, scale);
        //cout << "wave_length " << endl << wave_length << endl;
        float fo = 1.0 / wave_length;
        MatrixXf mat_temp = ((radius / fo).unaryExpr(mylog<float>())).cwiseAbs2();
        mat_temp = -mat_temp / (2 * pow(log(sigma__onf), 2)); //op
        logGabor = mat_temp.unaryExpr(myexpo<float>());
        logGabor = (logGabor).binaryExpr(lp, mymulpli<float>());
        logGabor(0, 0) = 0;

        MatrixXcf IMF = IM.binaryExpr(logGabor, mymulpli_complex_float<std::complex<float>>());
        //cout << "IMF " << endl;
        //cout << IMF << endl;
        fftwf_plan m_invplan;
        fftwf_complex *m_imgfftConv;
        m_imgfftConv = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * (nrows*ncols));
        m_invplan = fftwf_plan_dft_2d(nrows, ncols, m_imgfftConv, m_imgfftConv, FFTW_BACKWARD, FFTW_ESTIMATE);
        eigen_matrix_to_complex_array(IMF, m_imgfftConv);
        fftwf_execute(m_invplan);
        MatrixXcf ifft_S = MatrixXcf::Zero(nrows, ncols);
        complex_array_eigen_matrix(ifft_S, m_imgfftConv);

        MatrixXf f = ifft_S.real() / (nrows*ncols);
        //cout << "f " << endl;
        //cout << f << endl;

        eigen_matrix_to_complex_array(IMF.binaryExpr(H, mymulpli_complex_complex<std::complex<float>>()), m_imgfftConv);
        fftwf_execute(m_invplan);

        MatrixXcf h = MatrixXcf::Zero(nrows, ncols);
        complex_array_eigen_matrix(h, m_imgfftConv);
        h = h / (nrows * ncols);
        //cout << "h " << endl;
        //cout << h << endl;
        MatrixXf h1 = h.real();
        MatrixXf h2 = h.imag();
        MatrixXf An = (f.cwiseAbs2() + h1.cwiseAbs2() + h2.cwiseAbs2()).cwiseSqrt();

        sumAn = sumAn + An;             //Sum of component amplitudes over scale.
        sumf = sumf + f;
        sumh1 = sumh1 + h1;
        sumh2 = sumh2 + h2;

        //At the smallest scale estimate noise characteristics from the
        // distribution of the filter amplitude responses stored in sumAn.
        // tau is the Rayleigh parameter that is used to describe the
        // distribution.

        if (scale == 0)
        {
            if (noiseMethod == -1) //Use median to estimate noise statistics
            {
                VectorXf temp_vec(Map<VectorXf>(sumAn.data(), sumAn.cols()*sumAn.rows()));
                tau = median_vector(temp_vec) / sqrt_log_4;
            }
            else if (noiseMethod == -2) // Use mode to estimate noise statistics
            {
                VectorXf temp_vec(Map<VectorXf>(sumAn.data(), sumAn.cols()*sumAn.rows()));
                tau = rayleigh_mode(temp_vec);
            }
            maxAn = An;
        }
        else
        {
            //Record maximum amplitude of components across scales.  This is needed
            //to determine the frequency spread weighting.
            maxAn = maxAn.cwiseMax(An);
        }
    }
    MatrixXf wdth = (sumAn.binaryExpr(maxAn + matrix_epsilon, mydivide<float>()) - MatrixXf::Ones(nrows, ncols)) / (nscale - 1);

    //cout << "wdth " << endl;
    //cout << wdth << endl;

    MatrixXf mat_cutoff = MatrixXf::Constant(nrows, ncols, cutt_off);
    MatrixXf mat_temp_ = MatrixXf::Ones(nrows, ncols) + (g*(mat_cutoff - wdth)).unaryExpr(myexpo<float>());
    MatrixXf weight = (MatrixXf::Ones(nrows, ncols)).binaryExpr(mat_temp_, mydivide<float>());

    if (T == -100)
    {
        if (noiseMethod >= 0) //We are using a fixed noise threshold
        {
            T = noiseMethod; //use supplied noiseMethod value as the threshold
        }
        else
        {
            // Estimate the effect of noise on the sum of the filter responses as
            // the sum of estimated individual responses(this is a simplistic
            // overestimate).As the estimated noise response at succesive scales
            // is scaled inversely proportional to bandwidth we have a simple
            // geometric sum.
            float totalTau = tau * (1 - pow(1 / mult, nscale)) / (1 - (1 / mult));
            //Calculate mean and std dev from tau using fixed relationship
            //between these parameters and tau.See
            float EstNoiseEnergyMean = totalTau*sqrt(M_PI / 2);        // Expected mean and std
            float EstNoiseEnergySigma = totalTau*sqrt((4 - M_PI) / 2);   // values of noise energy
            T = EstNoiseEnergyMean + k*EstNoiseEnergySigma; // Noise threshold
        }
    }

    // Apply noise threshold, this is effectively wavelet denoising via
    // soft thresholding.
    MatrixXf mat_T = MatrixXf::Constant(nrows, ncols, T);

    orient = (-sumh2.binaryExpr(sumh1, mydivide<float>())).unaryExpr(my_atan<float>());
    orient = orient.unaryExpr(my_wrap_angle_0_pi<float>());
    orient = (180 * orient / M_PI).unaryExpr(my_fix<float>());

    //cout << "orient " << endl;
    //cout << orient << endl;

    ft = sumf.binaryExpr((sumh1.cwiseAbs2() + sumh2.cwiseAbs2()).cwiseSqrt(), my_atan2<float>());
    //cout << "ft " << endl;
    //cout << ft << endl;

    MatrixXf energy = (sumf.cwiseAbs2() + sumh1.cwiseAbs2() + sumh2.cwiseAbs2()).cwiseSqrt();

    //cout << "energy " << endl;
    //cout << energy << endl;

    //Compute phase congruency.The original measure,
    //PC = energy / sumAn
    //is proportional to the weighted cos(phasedeviation).This is not very
    //localised so this was modified to
    //PC = cos(phasedeviation) - | sin(phasedeviation) |
    //(Note this was actually calculated via dot and cross products.)  This measure
    //approximates
    //PC = 1 - phasedeviation.

    //However, rather than use dot and cross products it is simpler and more
    //efficient to simply use acos(energy / sumAn) to obtain the weighted phase
    //deviation directly.Note, in the expression below the noise threshold is
    //not subtracted from energy immediately as this would interfere with the
    //phase deviation computation.Instead it is applied as a weighting as a
    //fraction by which energy exceeds the noise threshold.This weighting is
    //applied in addition to the weighting for frequency spread.Note also the
    //phase deviation gain factor which acts to sharpen up the edge response.A
    //value of 1.5 seems to work well.Sensible values are from 1 to about 2.

    MatrixXf temp_mat = sumAn + matrix_epsilon;
    temp_mat = energy.binaryExpr(temp_mat, mydivide<float>());
    temp_mat = MatrixXf::Ones(nrows, ncols) - deviation_gain * temp_mat.unaryExpr(my_acos<float>());
    temp_mat = temp_mat.cwiseMax(MatrixXf::Zero(nrows, ncols));
    MatrixXf temp_mat1 = (energy - mat_T).cwiseMax(MatrixXf::Zero(nrows, ncols));

    temp_mat1 = temp_mat1.binaryExpr((energy + matrix_epsilon), mydivide<float>());
    temp_mat = weight.binaryExpr(temp_mat, mymulpli<float>());
    PC = temp_mat.binaryExpr(temp_mat1, mymulpli<float>());

    //cout << "PC " << endl;
    //cout << PC << endl;

    /*//get location of maximum
    MatrixXf::Index maxRow, maxCol;
    float max = PC.maxCoeff(&maxRow, &maxCol);
    //get location of minimum
    MatrixXf::Index minRow, minCol;
    float min = PC.minCoeff(&minRow, &minCol);

    /*cout << "Max: " << max << ", at: " <<
        maxRow << "," << maxCol << endl;
    cout << "Min: " << min << ", at: " <<
        minRow << "," << minCol << endl;*

        //cout << " threshold " << T << endl;
    for (int row = 0; row < nrows; row++)
    {
        for (int col = 0; col < ncols; col++)
        {
            //PC(row, col) = int(PC(row, col) * 255 / max);
            //cout << PC(row, col) << endl;
        }
    }/**/
}

//yannick.zoetgnande@etudiant.univ-rennes1.fr

void phasecongmono(cv::Mat img_, int nscale, int norient, float min_wave_length,
    float mult, float sigma__onf, float k, float cutt_off, int noiseMethod, float g, float deviation_gain, MatrixXf &PC, MatrixXf &ft, float &T, MatrixXf &orient)
{
    MatrixXf img = mat_to_eigen(img_);
    double sqrt_log_4 = sqrt(log(4));
    //MatrixXf img(4, 3);
    //img << 1, 2, 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25;
    //img << 1, 23, 45, 41, 85, 23, 1, 2, 3, 7, 8, 9;
    float epsilon = 0.0001;
    int nrows = img.rows(), ncols = img.cols();
    MatrixXcf IM, S;
    MatrixXf p, s;
    perfft2(IM, S, p, s, img);

    MatrixXf sumAn = MatrixXf::Zero(nrows, ncols);
    MatrixXf sumf = MatrixXf::Zero(nrows, ncols);
    MatrixXf sumh1 = MatrixXf::Zero(nrows, ncols);
    MatrixXf sumh2 = MatrixXf::Zero(nrows, ncols);
    MatrixXf radius, u1, u2;
    filter_grid(nrows, ncols, radius, u1, u2);
    radius(0, 0) = 1;
    const std::complex<float> If(0.0f, 1.0f);
    MatrixXcf u1u2 = MatrixXcf::Zero(nrows, ncols);
    u1u2 = -u2 + u1*If;
    //u1u2 = u1u2.binaryExpr(-u2, set_real_part());
    //u1u2 = u1u2.binaryExpr(u1, set_imag_part());
    MatrixXcf H = u1u2.binaryExpr(radius, mydivid_complex_float<std::complex<float>>());
    //cout << "H " << endl;
    //cout << H << endl;
    MatrixXf lp = low_pass_filter(nrows, ncols, 0.45, 15);
    MatrixXf logGabor;
    MatrixXf maxAn;
    MatrixXf matrix_two = MatrixXf::Constant(nrows, ncols, 2);
    MatrixXf matrix_pi = MatrixXf::Constant(nrows, ncols, M_PI);
    MatrixXf matrix_epsilon = MatrixXf::Constant(nrows, ncols, epsilon);
    float tau;
    for (int scale = 0; scale < nscale; scale++)
    {
        float wave_length = min_wave_length * pow(mult, scale);
        //cout << "wave_length " << endl << wave_length << endl;
        float fo = 1.0 / wave_length;
        MatrixXf mat_temp = ((radius / fo).unaryExpr(mylog<float>())).cwiseAbs2();
        mat_temp = -mat_temp / (2 * pow(log(sigma__onf), 2));
        logGabor = mat_temp.unaryExpr(myexpo<float>());
        logGabor = (logGabor).binaryExpr(lp, mymulpli<float>());
        logGabor(0, 0) = 0;

        MatrixXcf IMF = IM.binaryExpr(logGabor, mymulpli_complex_float<std::complex<float>>());
        //cout << "IMF " << endl;
        //cout << IMF << endl;
        fftwf_plan m_invplan;
        fftwf_complex *m_imgfftConv;
        m_imgfftConv = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * (nrows*ncols));
        m_invplan = fftwf_plan_dft_2d(nrows, ncols, m_imgfftConv, m_imgfftConv, FFTW_BACKWARD, FFTW_ESTIMATE);
        eigen_matrix_to_complex_array(IMF, m_imgfftConv);
        fftwf_execute(m_invplan);
        MatrixXcf ifft_S = MatrixXcf::Zero(nrows, ncols);
        complex_array_eigen_matrix(ifft_S, m_imgfftConv);

        MatrixXf f = ifft_S.real() / (nrows*ncols);
        //cout << "f " << endl;
        //cout << f << endl;

        eigen_matrix_to_complex_array(IMF.binaryExpr(H, mymulpli_complex_complex<std::complex<float>>()), m_imgfftConv);
        fftwf_execute(m_invplan);

        MatrixXcf h = MatrixXcf::Zero(nrows, ncols);
        complex_array_eigen_matrix(h, m_imgfftConv);
        h = h / (nrows * ncols);
        //cout << "h " << endl;
        //cout << h << endl;
        MatrixXf h1 = h.real();
        MatrixXf h2 = h.imag();
        MatrixXf An = (f.cwiseAbs2() + h1.cwiseAbs2() + h2.cwiseAbs2()).cwiseSqrt();

        sumAn = sumAn + An;             //Sum of component amplitudes over scale.
        sumf = sumf + f;
        sumh1 = sumh1 + h1;
        sumh2 = sumh2 + h2;

        //At the smallest scale estimate noise characteristics from the
        // distribution of the filter amplitude responses stored in sumAn.
        // tau is the Rayleigh parameter that is used to describe the
        // distribution.

        if (scale == 0)
        {
            if (noiseMethod == -1) //Use median to estimate noise statistics
            {
                VectorXf temp_vec(Map<VectorXf>(sumAn.data(), sumAn.cols()*sumAn.rows()));
                tau = median_vector(temp_vec) / sqrt_log_4;
            }
            else if (noiseMethod == -2) // Use mode to estimate noise statistics
            {
                VectorXf temp_vec(Map<VectorXf>(sumAn.data(), sumAn.cols()*sumAn.rows()));
                tau = rayleigh_mode(temp_vec);
            }
            maxAn = An;
        }
        else
        {
            //Record maximum amplitude of components across scales.  This is needed
            //to determine the frequency spread weighting.
            maxAn = maxAn.cwiseMax(An);
        }
    }
    MatrixXf wdth = (sumAn.binaryExpr(maxAn + matrix_epsilon, mydivide<float>()) - MatrixXf::Ones(nrows, ncols)) / (nscale - 1);

    //cout << "wdth " << endl;
    //cout << wdth << endl;

    MatrixXf mat_cutoff = MatrixXf::Constant(nrows, ncols, cutt_off);
    MatrixXf mat_temp_ = MatrixXf::Ones(nrows, ncols) + (g*(mat_cutoff - wdth)).unaryExpr(myexpo<float>());
    MatrixXf weight = (MatrixXf::Ones(nrows, ncols)).binaryExpr(mat_temp_, mydivide<float>());

    if (T == -100)
    {
        if (noiseMethod >= 0) //We are using a fixed noise threshold
        {
            T = noiseMethod; //use supplied noiseMethod value as the threshold
        }
        else
        {
            // Estimate the effect of noise on the sum of the filter responses as
            // the sum of estimated individual responses(this is a simplistic
            // overestimate).As the estimated noise response at succesive scales
            // is scaled inversely proportional to bandwidth we have a simple
            // geometric sum.
            float totalTau = tau * (1 - pow(1 / mult, nscale)) / (1 - (1 / mult));
            //Calculate mean and std dev from tau using fixed relationship
            //between these parameters and tau.See
            float EstNoiseEnergyMean = totalTau*sqrt(M_PI / 2);        // Expected mean and std
            float EstNoiseEnergySigma = totalTau*sqrt((4 - M_PI) / 2);   // values of noise energy
            T = EstNoiseEnergyMean + k*EstNoiseEnergySigma; // Noise threshold
        }
    }

    // Apply noise threshold, this is effectively wavelet denoising via
    // soft thresholding.
    MatrixXf mat_T = MatrixXf::Constant(nrows, ncols, T);

    orient = (-sumh2.binaryExpr(sumh1, mydivide<float>())).unaryExpr(my_atan<float>());
    orient = orient.unaryExpr(my_wrap_angle_0_pi<float>());
    orient = (180 * orient / M_PI).unaryExpr(my_fix<float>());

    //cout << "orient " << endl;
    //cout << orient << endl;

    ft = sumf.binaryExpr((sumh1.cwiseAbs2() + sumh2.cwiseAbs2()).cwiseSqrt(), my_atan2<float>());
    //cout << "ft " << endl;
    //cout << ft << endl;

    MatrixXf energy = (sumf.cwiseAbs2() + sumh1.cwiseAbs2() + sumh2.cwiseAbs2()).cwiseSqrt();

    //cout << "energy " << endl;
    //cout << energy << endl;

    //Compute phase congruency.The original measure,
    //PC = energy / sumAn
    //is proportional to the weighted cos(phasedeviation).This is not very
    //localised so this was modified to
    //PC = cos(phasedeviation) - | sin(phasedeviation) |
    //(Note this was actually calculated via dot and cross products.)  This measure
    //approximates
    //PC = 1 - phasedeviation.

    //However, rather than use dot and cross products it is simpler and more
    //efficient to simply use acos(energy / sumAn) to obtain the weighted phase
    //deviation directly.Note, in the expression below the noise threshold is
    //not subtracted from energy immediately as this would interfere with the
    //phase deviation computation.Instead it is applied as a weighting as a
    //fraction by which energy exceeds the noise threshold.This weighting is
    //applied in addition to the weighting for frequency spread.Note also the
    //phase deviation gain factor which acts to sharpen up the edge response.A
    //value of 1.5 seems to work well.Sensible values are from 1 to about 2.

    MatrixXf temp_mat = sumAn + matrix_epsilon;
    temp_mat = energy.binaryExpr(temp_mat, mydivide<float>());
    temp_mat = MatrixXf::Ones(nrows, ncols) - deviation_gain * temp_mat.unaryExpr(my_acos<float>());
    temp_mat = temp_mat.cwiseMax(MatrixXf::Zero(nrows, ncols));
    MatrixXf temp_mat1 = (energy - mat_T).cwiseMax(MatrixXf::Zero(nrows, ncols));

    temp_mat1 = temp_mat1.binaryExpr((energy + matrix_epsilon), mydivide<float>());
    temp_mat = weight.binaryExpr(temp_mat, mymulpli<float>());
    PC = temp_mat.binaryExpr(temp_mat1, mymulpli<float>());



}

void non_max_sup(MatrixXf img_in, MatrixXf orient, float radius, MatrixXf &img_out, MatrixXcf &location)
{
    int nrows = img_in.rows(), ncols = img_in.cols();
    assert(img_in.cols() == orient.cols() && img_in.rows() == orient.rows() && "The image and the orient matrices must have the same size");
    assert(radius > 1 && "radius must be greater than 1");
    img_out = MatrixXf::Zero(nrows, ncols);
    location = MatrixXcf::Zero(nrows, ncols);

    int iradius = ceil(radius);
    VectorXf angle(181);
    for (int i = 0; i <= 180; i++)
    {
        angle[i] = i*M_PI / 180;
    }

    cout << " non_max_sup 1" << endl;

    VectorXf xoff = radius*angle.unaryExpr(mycosine<float>());
    VectorXf yoff = radius*angle.unaryExpr(mysine<float>());

    VectorXf hfrac = xoff - xoff.unaryExpr(myfloor<float>());
    VectorXf vfrac = yoff - yoff.unaryExpr(myfloor<float>());

    //orient = fix(orient)+1;

    cout << " non_max_sup 2" << endl;

    for (int row = iradius; row < nrows - iradius; row++)
    {
        for (int col = iradius; col < ncols - iradius; col++)
        {

            //cout << "non_max_sup " << row << "  " << col << "  " << iradius << endl;

            //cout << "non_max_sup a  " << row << "  " << col << "  " << iradius << endl;

            float or_ = orient(row, col);
            float x = col + xoff(or_);
            float y = row - yoff(or_);

            int fx = floor(x);
            int cx = ceil(x);
            int fy = floor(y);
            int cy = ceil(y);

            float tl = img_in(fy, fx);
            float tr = img_in(fy, cx);
            float bl = img_in(cy, fx);
            float br = img_in(cy, cx);

            float upperavg = tl + hfrac(or_)* (tr - tl);
            float loweravg = bl + hfrac(or_) * (br - bl);
            float v1 = upperavg + vfrac(or_) * (loweravg - upperavg);
            if (img_in(row, col) > v1)
            {
                x = col - xoff(or_);
                y = row + yoff(or_);

                fx = floor(x);
                cx = ceil(x);
                fy = floor(y);
                cy = ceil(y);
                tl = img_in(fy, fx);    //Value at top left integer pixel location.
                tr = img_in(fy, cx);    // top right
                bl = img_in(cy, fx);    // bottom left
                br = img_in(cy, cx);    // bottom right
                upperavg = tl + hfrac(or_) * (tr - tl);
                loweravg = bl + hfrac(or_) * (br - bl);
                float v2 = upperavg + vfrac(or_) * (loweravg - upperavg);
                //cout << "here " << endl;
                if (img_in(row, col) > v2)
                {
                    img_out(row, col) = img_in(row, col);

                    float c = img_in(row,col);
                    float a = (v1 + v2)/2 - c;
                    float b = a + c - v1;

                    float r = -b/(2*a);
                    location(row,col) = std::complex<float>(row + r*yoff(or_) , col + r*xoff(or_)) ;/**/
                    //cout << "Val " << location(row,col) << endl;
                }/**/
            }
            //cout << "non_max_sup b  " << row << "  " << col << "  " << iradius << endl;
        }
    }

    cout << " non_max_sup 3" << endl;

    /*MatrixXf squeleton = MatrixXf::Zero(nrows, ncols);
    thinning(img_out, squeleton);
    img_out = img_out.cwiseProduct(squeleton);
    location = location.cwiseProduct(squeleton);/**/

}

}
