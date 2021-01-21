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
#include "core.hpp"

#define  FFTW_IN_PLACE
#pragma comment(lib, "fftw3f")

#include <time.h>
#include <chrono>

using namespace std;
 
using namespace Eigen;
using namespace std::chrono;

#ifdef vpp
#include <vpp/vpp.hh>
#include <vpp/utils/opencv_bridge.hh>
#include <vpp/utils/opencv_utils.hh>
using namespace vpp;
#endif

namespace tpp {

    template <typename Type>
    struct phase_congruency
    {
        Type k;
        int norient;
        int nscale;
        Type min_wave_length;
        Type cutt_off;
        Type mult;
        Type sigma__onf;
        Type g;
        int noiseMethod;
        Type epsilon;
        phase_congruency()
        {
            nscale = 4;
            norient = 6;
            sigma__onf = 0.55;
            mult = 2.1;
            min_wave_length = 3;
            cutt_off = 0.5;
            noiseMethod = -1;
            g = 10;
            k = 2;
            epsilon = 0.0001;
        }
    };

#ifdef vpp
    template <typename Type>
    void display_image2d(image2d<Type> img)
    {
        for(int row = 0 ; row < img.nrows(); row ++)
        {
            for(int col = 0; col < img.ncols(); col++)
            {
                cout << img(row, col) << "  ";
            }
            cout << endl;
        }
        cout << endl << endl;
    }

    template <typename Type>
    image2d<Type> initialise_image2d(int nrows, int ncols, Type value)
    {
        image2d<Type> img(nrows, ncols);
        pixel_wise(img) | [&] (auto& z) { z = value; };
        return img;
    }
#endif

#ifdef vpp
    template <typename Type>
    struct phase_congruency_output_vpp
    {
        image2d<Type> M;
        image2d<Type> m;
        image2d<Type> orientation;
        image2d<Type> featType;
        std::vector<image2d<Type>> PC;
        image2d<Type> pcSum;
        std::vector<image2d<std::complex<Type>>> EO;
        Type T;

        int nrows;
        int ncols;
        int norient;
        int nscale;


        phase_congruency<Type> pc_settings;
        image2d<Type> matrix_two;
        image2d<Type> matrix_pi;
        image2d<Type> matrix_epsilon;
        std::vector<image2d<Type>> logGabor;
        image2d<Type> sin_theta;
        image2d<Type> cos_theta;
        std::vector<image2d<Type>> list_spread;

        phase_congruency_output_vpp()
        {
            T = 100;
        }

        phase_congruency_output_vpp(int nr, int nc, phase_congruency<Type> pc_set)
        {
            pc_settings = pc_set;
            T = 100;
            nrows = nr;
            ncols = nc;
            norient = pc_settings.norient;
            nscale = pc_settings.nscale;

            logGabor.resize(nscale);
            //cout << logGabor.size() << endl;
            sin_theta = image2d<Type>(nrows, ncols);
            cos_theta = image2d<Type>(nrows, ncols);


            matrix_two = image2d<Type>(nrows, ncols);
            pixel_wise(matrix_two) | [&] (auto& m) { m = Type(2); };

            matrix_pi = image2d<Type>(nrows, ncols);
            pixel_wise(matrix_pi) | [&] (auto& m) { m = Type(M_PI); };

            matrix_epsilon = image2d<Type>(nrows, ncols);
            pixel_wise(matrix_epsilon) | [&] (auto& m) { m = Type(pc_settings.epsilon); };

        }

        void initialize_for_pc()
        {
            VectorXf xrange;
            VectorXf yrange;
            if (ncols % 2 == 0)
            {
                xrange = range_number_pair(ncols) / ncols;
            }
            else
            {
                xrange = range_number_impair(ncols) / ncols;
            }

            if (nrows % 2 == 0)
            {
                yrange = range_number_pair(nrows) / nrows;
            }
            else
            {
                yrange = range_number_impair(nrows) / nrows;
            }


            MatrixXf x, y;
            mesgrid(xrange, yrange, x, y);

            MatrixXf radius = (x.cwiseAbs2() + y.cwiseAbs2()).cwiseSqrt();
            MatrixXf theta = -y.binaryExpr(x, my_atan2<float>());


            radius = ifftshift_matrix(radius);
            theta = ifftshift_matrix(theta);
            radius(0, 0) = 1;

            //sin_theta = image2d<Type>(theta.rows(), theta.cols());
            //cos_theta = image2d<Type>(theta.rows(), theta.cols());

            pixel_wise(sin_theta, cos_theta, cos_theta.domain()) | [&] (auto& st, auto &ct, auto coord)
            {
                int row = coord[0];
                int col = coord[1];
                st = sin(theta(row, col));
                ct = cos(theta(row, col));
            };

            MatrixXf lp = low_pass_filter(nrows, ncols, 0.45, 15);

            for (int scale = 0; scale < nscale; scale++)
            {
                logGabor[scale] = image2d<Type>(nrows, ncols);
                float wave_length = pc_settings.min_wave_length * pow(pc_settings.mult, scale);
                float fo = 1.0 / wave_length;
                MatrixXf mat_temp = ((radius / fo).unaryExpr(mylog<float>())).cwiseAbs2();
                mat_temp = -mat_temp / (2 * pow(log(pc_settings.sigma__onf), 2));
                pixel_wise(logGabor[scale], logGabor[scale].domain()) | [&] (auto& v, auto coord)
                {
                    int row = coord[0];
                    int col = coord[1];
                    v = exp(mat_temp(row, col)) * lp(row, col);
                };
                logGabor[scale](0, 0) = 0;

            }

            for (int orient = 0; orient < norient; orient++)
            {
                Type angl = orient * M_PI / norient; //Filter angle.
                image2d<Type> diff_sine(nrows, ncols);
                image2d<Type> diff_cosine(nrows, ncols);
                image2d<Type> dtheta(nrows, ncols);

                pixel_wise(sin_theta, cos_theta, diff_sine, diff_cosine) | [&]
                        (auto sin_theta, auto cos_theta, auto& ds, auto &dc)
                {
                    ds = sin_theta * cos(angl) - cos_theta * sin(angl);
                    dc = cos_theta *cos(angl) + sin_theta * sin(angl);
                };


                pixel_wise(dtheta, diff_sine, diff_cosine) | [&] (auto &dt, auto ds, auto dc)
                {
                    dt = min(Type(fabs(atan2(ds, dc)) * norient / 2), Type(M_PI));
                };

                image2d<Type> spread(nrows, ncols);
                pixel_wise(spread, dtheta) | [&] (auto &sp, auto dt)
                {
                    sp = (cos(dt) + 1)/2;
                };

                list_spread.push_back(spread);
            }

        }

        void initialize_moments()
        {

            M = initialise_image2d(nrows, ncols, Type(0));
            m = initialise_image2d(nrows, ncols, Type(0));
            orientation = initialise_image2d(nrows, ncols, Type(0));
            featType = initialise_image2d(nrows, ncols, Type(0));
            pcSum = initialise_image2d(nrows, ncols, Type(0));

            //Array of convolution results
            EO.resize(nscale * norient);
            PC.resize(norient);

            int i;
            for (int orient = 0; orient < norient; orient++)
            {
                for (int scale = 0; scale < nscale; scale++)
                {
                    i = orient * nscale + scale;
                    EO[i] = initialise_image2d(nrows, ncols, std::complex<Type>(0,0));
                    //EO[i] = image2d<std::complex<Type>>(nrows, ncols);
                }
                PC[orient] = initialise_image2d(nrows, ncols, Type(0));
            }
        }

    };
#endif


    template <typename Type>
    struct phase_congruency_output_eigen
    {
        Matrix<Type, Dynamic, Dynamic> M;
        Matrix<Type, Dynamic, Dynamic> m;
        Matrix<Type, Dynamic, Dynamic> orientation;
        Matrix<Type, Dynamic, Dynamic> featType;
        std::vector<Matrix<Type, Dynamic, Dynamic>> PC;
        Matrix<Type, Dynamic, Dynamic> pcSum;
        std::vector<Matrix<std::complex<Type>, Dynamic, Dynamic>> EO;
        Type T;

        int nrows;
        int ncols;
        int norient;
        int nscale;


        phase_congruency<Type> pc_settings;
        Matrix<Type, Dynamic, Dynamic> matrix_two;
        Matrix<Type, Dynamic, Dynamic> matrix_pi;
        Matrix<Type, Dynamic, Dynamic> matrix_epsilon;
        std::vector<Matrix<Type, Dynamic, Dynamic>> logGabor;
        Matrix<Type, Dynamic, Dynamic> sin_theta;
        Matrix<Type, Dynamic, Dynamic> cos_theta;
        std::vector<Matrix<Type, Dynamic, Dynamic>> list_spread;

        phase_congruency_output_eigen()
        {
            T = 100;
        }

        phase_congruency_output_eigen(int nr, int nc, phase_congruency<Type> pc_set)
        {
            pc_settings = pc_set;
            T = 100;
            nrows = nr;
            ncols = nc;
            norient = pc_settings.norient;
            nscale = pc_settings.nscale;

            logGabor.resize(nscale);
            //cout << logGabor.size() << endl;
            sin_theta = Matrix<Type, Dynamic, Dynamic>(nrows, ncols);
            cos_theta = Matrix<Type, Dynamic, Dynamic>(nrows, ncols);

            matrix_two = Matrix<Type, Dynamic, Dynamic>::Constant(nrows, ncols, Type(2));
            matrix_pi = Matrix<Type, Dynamic, Dynamic>::Constant(nrows, ncols, Type(M_PI));
            matrix_epsilon = Matrix<Type, Dynamic, Dynamic>::Constant(nrows, ncols, Type(pc_settings.epsilon));

            initialize_for_pc();
            initialize_moments();
        }

        void initialize_for_pc()
        {
            VectorXf xrange;
            VectorXf yrange;
            if (ncols % 2 == 0)
            {
                xrange = range_number_pair(ncols) / ncols;
            }
            else
            {
                xrange = range_number_impair(ncols) / ncols;
            }

            if (nrows % 2 == 0)
            {
                yrange = range_number_pair(nrows) / nrows;
            }
            else
            {
                yrange = range_number_impair(nrows) / nrows;
            }


            MatrixXf x, y;
            mesgrid(xrange, yrange, x, y);

            MatrixXf radius = (x.cwiseAbs2() + y.cwiseAbs2()).cwiseSqrt();
            MatrixXf theta = -y.binaryExpr(x, my_atan2<float>());


            radius = ifftshift_matrix(radius);
            theta = ifftshift_matrix(theta);
            radius(0, 0) = 1;

            #pragma omp parallel for
            for(int row = 0; row < nrows; row++)
            {
                for(int col = 0; col < ncols; col++)
                {
                    sin_theta(row, col) = sin(theta(row, col));
                    cos_theta(row, col) = cos(theta(row, col));
                }
            }



            MatrixXf lp = low_pass_filter(nrows, ncols, 0.45, 15);

            for (int scale = 0; scale < nscale; scale++)
            {
                logGabor[scale] = Matrix<Type, Dynamic, Dynamic>(nrows, ncols);
                float wave_length = pc_settings.min_wave_length * pow(pc_settings.mult, scale);
                float fo = 1.0 / wave_length;
                MatrixXf mat_temp = ((radius / fo).unaryExpr(mylog<float>())).cwiseAbs2();
                mat_temp = -mat_temp / (2 * pow(log(pc_settings.sigma__onf), 2));

                #pragma omp parallel for
                for(int row = 0; row < nrows; row++)
                {
                    for(int col = 0; col < ncols; col++)
                    {
                        logGabor[scale](row, col) = exp(mat_temp(row, col)) * lp(row, col);
                    }
                }
                logGabor[scale](0, 0) = 0;

            }

            for (int orient = 0; orient < norient; orient++)
            {
                Type angl = orient * M_PI / norient; //Filter angle.
                Matrix<Type, Dynamic, Dynamic> diff_sine(nrows, ncols);
                Matrix<Type, Dynamic, Dynamic> diff_cosine(nrows, ncols);
                Matrix<Type, Dynamic, Dynamic> dtheta(nrows, ncols);

                #pragma omp parallel for
                for(int row = 0; row < nrows; row++)
                {
                    for(int col = 0; col < ncols; col++)
                    {
                        diff_sine(row, col) = sin_theta(row, col) * cos(angl) - cos_theta(row, col) * sin(angl);
                        diff_cosine(row, col) = cos_theta(row, col) *cos(angl) + sin_theta(row, col) * sin(angl);
                    }
                }

                #pragma omp parallel for
                for(int row = 0; row < nrows; row++)
                {
                    for(int col = 0; col < ncols; col++)
                    {
                        dtheta(row, col) = min(Type(fabs(atan2(diff_sine(row, col), diff_cosine(row, col))) * norient / 2), Type(M_PI));
                    }
                }


                Matrix<Type, Dynamic, Dynamic> spread(nrows, ncols);

                #pragma omp parallel for
                for(int row = 0; row < nrows; row++)
                {
                    for(int col = 0; col < ncols; col++)
                    {
                        spread(row, col) = (cos(dtheta(row, col)) + 1)/2;
                    }
                }

                list_spread.push_back(spread);
            }

        }

        inline
        void initialize_moments()
        {

            M = Matrix<Type, Dynamic, Dynamic>::Zero(nrows, ncols);
            m = Matrix<Type, Dynamic, Dynamic>::Zero(nrows, ncols);
            orientation = Matrix<Type, Dynamic, Dynamic>::Zero(nrows, ncols);
            featType = Matrix<Type, Dynamic, Dynamic>::Zero(nrows, ncols);
            pcSum = Matrix<Type, Dynamic, Dynamic>::Zero(nrows, ncols);

            //Array of convolution results
            EO.resize(nscale * norient);
            PC.resize(norient);

            int i;
            for (int orient = 0; orient < norient; orient++)
            {
                for (int scale = 0; scale < nscale; scale++)
                {
                    i = orient * nscale + scale;
                    //EO[i] = Matrix<std::complex<Type>, Dynamic, Dynamic>(nrows, ncols, std::complex<Type>(0, 0));
                    EO[i] = Matrix<std::complex<Type>, Dynamic, Dynamic>::Zero(nrows, ncols);
                }
                PC[orient] = Matrix<Type, Dynamic, Dynamic>::Zero(nrows, ncols);
            }
        }

    };


}
