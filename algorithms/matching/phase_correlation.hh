#pragma once


#define _USE_MATH_DEFINES
//adapted from https://github.com/amarburg/opencv-correlation/blob/master/lib/phase_correlation.c

#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>


#include "algorithms/matching.hh"
#include "algorithms/morphology.hh"
#include "algorithms/segmentation.hh"
#include "algorithms/miscellaneous.hh"

#include <omp.h>

using namespace std;
using namespace Eigen;


namespace tpp
{

void rectangular_low_pass(MatrixXf &H1,int U1, int U2,int m)
{
    for(int row = 0 ; row < H1.rows() ; row++)
    {
        for(int col = 0 ; col < H1.cols() ; col++)
        {
            int k1 = row - m;
            int k2 = col - m;
            if(abs(k1) <= U1 && abs(k2) <= U2 )
            H1(row,col) = 1;
            else
                H1(row,col) = 0;
        }
    }
}

void rectangular_low_pass_H2(MatrixXf &H1,int U1, int U2,int m)
{
    for(int row = 0 ; row < H1.rows() ; row++)
    {
        for(int col = 0 ; col < H1.cols() ; col++)
        {
            int k1 = row - m;
            int k2 = col - m;
            if(abs(k1) <= U1 && abs(k2) <= U2 )
            H1(row,col) = 1;
            else
                H1(row,col) = 0;
        }
    }
}


void rectangular_low_pass_H3(MatrixXf &H1,int U1, int U2,int m)
{
    for(int row = 0 ; row < H1.rows() ; row++)
    {
        for(int col = 0 ; col < H1.cols() ; col++)
        {
            int k1 = row - m;
            int k2 = col - m;
            if(abs(k1) <= U1 && abs(k2) <= U2 )
            H1(row,col) = 1;
            else
                H1(row,col) = 0;
        }
    }
}



inline
MatrixXf hanning_window(MatrixXf img,int M)
{
    int nrows = img.rows();
    int ncols = img.cols();
    MatrixXf res = MatrixXf::Zero(nrows,ncols);
    double omega_r = M_PI/nrows;
    double omega_c = M_PI/ncols;
    for(int n1 = -M ; n1 < nrows-M; n1++)
    {
        int row = n1+M;
        for(int n2 = -M ; n2 < ncols-M ; n2++)
        {
            int col = n2+M;
            //cout << n1 << " " << n2 << "       ";
            res(row,col) = 0.25 * ( 1 + cos(omega_r*n1)) * ( 1 + cos(omega_c*n2));
        }
        //cout << endl;
    }
    //cout << endl << endl << endl;
    return res;
}

inline
MatrixXf hanning_window(MatrixXf img)
{
    int nrows = img.rows();
    int ncols = img.cols();
    MatrixXf res = MatrixXf::Zero(nrows,ncols);
    double omega_r = M_PI/nrows;
    double omega_c = M_PI/ncols;
    for(int n1 =0 ; n1 < nrows; n1++)
    {
        for(int n2 = 0 ; n2 < ncols ; n2++)
        {
            res(n1,n2) = 0.25 * ( 1 + cos(omega_r*n1)) * ( 1 + cos(omega_c*n2));
        }
    }
    return res;
}

inline
void hanning_window_(MatrixXf &img)
{
    int nrows = img.rows();
    int ncols = img.cols();
    int fft_size = nrows * ncols;
    double omega_r = M_PI/nrows;
    double omega_c = M_PI/ncols;
    for(int row = 0 ; row < nrows; row++)
    {
        for(int col = 0 ; col < ncols ; col++)
        {
            /*float ru=2*u/M-1;
            float  rv = 2*v/N-1;
            float ruv=pow(ru,2)+pow(rv,2);
            ruv = sqrt(ruv);
            float wuv =  0.5*(cos(PI*ruv)+1);
*/
            img(row,col) = 0.25 * ( 1 + cos(omega_r*row)) * ( 1 + cos(omega_c*col));
        }
    }
}


MatrixXf phase_correlation_1d(MatrixXf mat1,MatrixXf mat2)
{
    int nrows = mat1.rows(),ncols = mat1.cols();
    //cout << "nrows " << nrows <<"  ncols " << ncols << endl;
    MatrixXf result = MatrixXf::Zero(nrows,ncols);

    double tmp;

    int fft_size = ncols *nrows;
    //cout << "fft_size " << fft_size << endl;

    // allocate FFTW input and output arrays 
    fftw_complex *img1 = ( fftw_complex* )fftw_malloc( sizeof( fftw_complex ) * fft_size);
    fftw_complex *img2 = ( fftw_complex* )fftw_malloc( sizeof( fftw_complex ) * fft_size );
    fftw_complex *res  = ( fftw_complex* )fftw_malloc( sizeof( fftw_complex ) * fft_size );

    // setup FFTW plans 
    fftw_plan fft_img1 = fftw_plan_dft_1d( fft_size, img1, img1, FFTW_FORWARD,  FFTW_ESTIMATE );
    fftw_plan fft_img2 = fftw_plan_dft_1d( fft_size, img2, img2, FFTW_FORWARD,  FFTW_ESTIMATE );
    fftw_plan ifft_res = fftw_plan_dft_1d( fft_size, res,  res,  FFTW_BACKWARD, FFTW_ESTIMATE );

    // load images' data to FFTW input 
    for(int row = 0, k = 0 ; row < nrows ; row++ ) {
        for(int col = 0 ; col < ncols ; col++, k++ ) {
            img1[k][0] = ( double )mat1(row,col);
            img1[k][1] = 0.0;

            img2[k][0] = ( double )mat2(row,col);
            img2[k][1] = 0.0;
        }
    }
    // obtain the FFT of img1 
    fftw_execute( fft_img1 );

    // obtain the FFT of img2 
    fftw_execute( fft_img2 );

    // obtain the cross power spectrum 
    for(int i = 0; i < fft_size ; i++ ) {
        res[i][0] = ( img2[i][0] * img1[i][0] ) - ( img2[i][1] * ( -img1[i][1] ) );
        res[i][1] = ( img2[i][0] * ( -img1[i][1] ) ) + ( img2[i][1] * img1[i][0] );

        tmp = sqrt( pow( res[i][0], 2.0 ) + pow( res[i][1], 2.0 ) );

        res[i][0] /= tmp;
        res[i][1] /= tmp;

        std::cout << "value " << res[i][0]  << "  "  << res[i][1]  << endl;
    }

    // obtain the phase correlation array 
    fftw_execute(ifft_res);

    // normalize and copy to result image 

    for(int row = 0, k = 0 ; row < nrows ; row++ ) {
        for(int col = 0 ; col < ncols ; col++, k++ ) {
            result(row,col) = res[k][0] / ( double )fft_size;
        }
    }

    //cout << "result " << result << endl;



    // deallocate FFTW arrays and plans 
    fftw_destroy_plan( fft_img1 );
    fftw_destroy_plan( fft_img2 );
    fftw_destroy_plan( ifft_res );
    fftw_free( img1 );
    fftw_free( img2 );
    fftw_free( res );



    return result;
}

MatrixXf phase_correlation(MatrixXf mat1,MatrixXf mat2)
{
    int M = (mat1.rows()-1)/2;
    //cout << "mat1 " << mat1 << endl;
    MatrixXf hann_ = hanning_window(mat1);
    //cout << "modified mat1 " << mat1 << endl;

    mat1 = mat1.cwiseProduct(hann_);
    mat2 = mat2.cwiseProduct(hann_);

    int nrows = mat1.rows(),ncols = mat1.cols();
    //cout << "nrows " << nrows <<"  ncols " << ncols << endl;
    MatrixXf result = MatrixXf::Zero(nrows,ncols);
    MatrixXf result_ord = MatrixXf::Zero(nrows,ncols);

    double tmp;

    int nw2 = ncols / 2 + 1;
    int fftwSz = nrows * nw2;

    float *mat1_array = (float *)malloc(sizeof(float) * (nrows*ncols));
    float *mat2_array = (float *)malloc(sizeof(float) * (nrows*ncols));

    eigen_matrix_to_float_array(mat1, mat1_array);
    eigen_matrix_to_float_array(mat2, mat2_array);/**/

    int fft_size = ncols *nrows;
    //cout << "fft_size " << fft_size << endl;

    /* allocate FFTW input and output arrays */
    fftwf_complex *mat1_fft = ( fftwf_complex* )fftw_malloc( sizeof( fftwf_complex ) * fftwSz);
    fftwf_complex *mat2_fft = ( fftwf_complex* )fftw_malloc( sizeof( fftwf_complex ) * fftwSz );
    fftwf_complex *res  = ( fftwf_complex* )fftw_malloc( sizeof( fftwf_complex ) * fft_size );
    MatrixXcf res_eigen = MatrixXcf::Zero(nrows, ncols);

    fftwf_complex *hermite_out_mat1 = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * (nrows*ncols));
    fftwf_complex *hermite_out_mat2 = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * (nrows*ncols));

    /* setup FFTW plans */
    fftwf_plan fft_img1 = fftwf_plan_dft_r2c_2d(nrows, ncols, mat1_array, mat1_fft, FFTW_ESTIMATE);
    fftwf_plan fft_img2 = fftwf_plan_dft_r2c_2d(nrows, ncols, mat2_array, mat2_fft, FFTW_ESTIMATE);
    fftwf_plan ifft_res = fftwf_plan_dft_2d(nrows, ncols, res, res, FFTW_BACKWARD, FFTW_ESTIMATE);


    /* obtain the FFT of img1 */
    fftwf_execute( fft_img1 );

    /* obtain the FFT of img2 */
    fftwf_execute( fft_img2 );

    hermite_to_array_pc(mat1_fft, hermite_out_mat1, ncols, nrows);
    hermite_to_array_pc(mat2_fft, hermite_out_mat2, ncols, nrows);


    /* obtain the cross power spectrum */
    for(int row = 0, i = 0 ; row < nrows ; row++ ) {
        for(int col = 0 ; col < ncols ; col++, i++ ) {
            res[i][0] = ( hermite_out_mat2[i][0] * hermite_out_mat1[i][0] ) - ( hermite_out_mat2[i][1] * ( -hermite_out_mat1[i][1] ) );
            res[i][1] = ( hermite_out_mat2[i][0] * ( -hermite_out_mat1[i][1] ) ) + ( hermite_out_mat2[i][1] * hermite_out_mat1[i][0] );

            tmp = sqrt( pow( res[i][0], 2.0 ) + pow( res[i][1], 2.0 ) );

            res[i][0] /= tmp;
            res[i][1] /= tmp;

        }

    }
    /* obtain the phase correlation array */
    fftwf_execute(ifft_res);

    /* normalize and copy to result image */

    for(int row = 0, k = 0 ; row < nrows ; row++ ) {
        for(int col = 0 ; col < ncols ; col++, k++ ) {
            result(row,col) = res[k][0] / ( float )fft_size;
        }
    }



    /* deallocate FFTW arrays and plans */
    fftwf_destroy_plan( fft_img1 );
    fftwf_destroy_plan( fft_img2 );
    fftwf_destroy_plan( ifft_res );
    fftwf_free( hermite_out_mat1 );
    fftwf_free( hermite_out_mat2 );
    fftwf_free( mat1_fft );
    fftwf_free( mat2_fft );
    fftwf_free( res );



    return result;
}


MatrixXf phase_correlation(MatrixXf mat1,MatrixXf mat2, MatrixXf H)
{
    int M = (mat1.rows()-1)/2;
    //cout << "mat1 " << mat1 << endl;
    MatrixXf hann_ = hanning_window(mat1);
    //cout << "modified mat1 " << mat1 << endl;

    mat1 = mat1.cwiseProduct(hann_);
    mat2 = mat2.cwiseProduct(hann_);

    int nrows = mat1.rows(),ncols = mat1.cols();
    //cout << "nrows " << nrows <<"  ncols " << ncols << endl;
    MatrixXf result = MatrixXf::Zero(nrows,ncols);

    double tmp;

    int nw2 = ncols / 2 + 1;
    int fftwSz = nrows * nw2;

    float *mat1_array = (float *)malloc(sizeof(float) * (nrows*ncols));
    float *mat2_array = (float *)malloc(sizeof(float) * (nrows*ncols));

    eigen_matrix_to_float_array(mat1, mat1_array);
    eigen_matrix_to_float_array(mat2, mat2_array);/**/

    int fft_size = ncols *nrows;
    //cout << "fft_size " << fft_size << endl;

    /* allocate FFTW input and output arrays */
    fftwf_complex *mat1_fft = ( fftwf_complex* )fftw_malloc( sizeof( fftwf_complex ) * fftwSz);
    fftwf_complex *mat2_fft = ( fftwf_complex* )fftw_malloc( sizeof( fftwf_complex ) * fftwSz );
    fftwf_complex *res  = ( fftwf_complex* )fftw_malloc( sizeof( fftwf_complex ) * fft_size );
    MatrixXcf res_eigen = MatrixXcf::Zero(nrows, ncols);

    fftwf_complex *hermite_out_mat1 = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * (nrows*ncols));
    fftwf_complex *hermite_out_mat2 = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * (nrows*ncols));

    /* setup FFTW plans */
    fftwf_plan fft_img1 = fftwf_plan_dft_r2c_2d(nrows, ncols, mat1_array, mat1_fft, FFTW_ESTIMATE);
    fftwf_plan fft_img2 = fftwf_plan_dft_r2c_2d(nrows, ncols, mat2_array, mat2_fft, FFTW_ESTIMATE);
    fftwf_plan ifft_res = fftwf_plan_dft_2d(nrows, ncols, res, res, FFTW_BACKWARD, FFTW_ESTIMATE);


    /* obtain the FFT of img1 */
    fftwf_execute( fft_img1 );

    /* obtain the FFT of img2 */
    fftwf_execute( fft_img2 );

    hermite_to_array_pc(mat1_fft, hermite_out_mat1, ncols, nrows);
    hermite_to_array_pc(mat2_fft, hermite_out_mat2, ncols, nrows);

    Eigen::MatrixXcf imga = MatrixXcf::Zero(nrows, ncols);;
    Eigen::MatrixXcf imgb = MatrixXcf::Zero(nrows, ncols);;

    complex_array_eigen_matrix(imga, hermite_out_mat1);
    complex_array_eigen_matrix(imgb, hermite_out_mat2);

    imga = fftshift_matrix(imga);
    imgb = fftshift_matrix(imgb);

    /* obtain the cross power spectrum */
    for(int row = 0, i = 0 ; row < nrows ; row++ ) {
        for(int col = 0 ; col < ncols ; col++, i++ ) {
            //for(int i = 0; i < fft_size ; i++ ) {
            res[i][0] = ( imgb(row,col).real() * imga(row,col).real() ) - ( imgb(row,col).imag() * ( -imga(row,col).imag() ) );
            res[i][1] = ( imgb(row,col).real() * ( -imga(row,col).imag() ) ) + ( imgb(row,col).imag() * imga(row,col).real() );

            tmp = sqrt( pow( res[i][0], 2.0 ) + pow( res[i][1], 2.0 ) );

            res[i][0] /= tmp;
            res[i][1] /= tmp;
        }
    }
    /* obtain the phase correlation array */
    fftwf_execute(ifft_res);

    /* normalize and copy to result image */

    for(int row = 0, k = 0 ; row < nrows ; row++ ) {
        for(int col = 0 ; col < ncols ; col++, k++ ) {
            result(row,col) = (res[k][0] / ( float )fft_size)*H(row,col);
        }
    }

    /* deallocate FFTW arrays and plans */
    fftwf_destroy_plan( fft_img1 );
    fftwf_destroy_plan( fft_img2 );
    fftwf_destroy_plan( ifft_res );
    fftwf_free( hermite_out_mat1 );
    fftwf_free( hermite_out_mat2 );
    fftwf_free( mat1_fft );
    fftwf_free( mat2_fft );
    fftwf_free( res );

    return result;
}



}
