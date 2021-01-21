#pragma once

#include <algorithm>
#include "gabor_util.hh"

#include <omp.h>

#ifdef vpp
#include "vpp/vpp.hh"
using namespace vpp;
#endif

namespace tpp
{


using namespace  std;
// input store by row-major order in should be outside initialized
void eigen_matrix_to_float_array(MatrixXf imga, float* in)
{
    int nw = imga.cols();
    int nh = imga.rows();
    int offset;
    #pragma omp parallel for
    for (int y = 0; y < nh; y++) // rows
    {
        for (int x = 0; x < nw; x++) // cols
        {
            in[x + y*nw] = imga(y, x);
        }
    }
}


#ifdef vpp
template< typename source, typename target>
void image2d_to_array(image2d<source> imga, target* in)
{
    int nw = imga.ncols();
    pixel_wise(imga, imga.domain()) | [&] (auto i, vint2 coord) {
        int x = coord[1];
        int y = coord[0];
        in[x + y*nw] = target(i);
    };
}

template< typename Type>
void image2d_to_complex_array(image2d<std::complex<Type>> imga, fftwf_complex* in)
{
    int nw = imga.ncols();
    pixel_wise(imga, imga.domain()) | [&] (auto i, vint2 coord) {
        int x = coord[1];
        int y = coord[0];
        in[x + y*nw][0] = i.real();
        in[x + y*nw][1] = i.imag();
    };
}

template< typename Type>
void complex_array_image2d(image2d<std::complex<Type>>& imga, fftwf_complex* in)
{
    int ncols = imga.ncols();
    int nrows = imga.nrows();
    //omp_set_num_threads(1);
    pixel_wise(imga, imga.domain()) | [&] (auto &i, vint2 coord) {
//#pragma omp critical
        {
        int x = coord[1];
        int y = coord[0];
        //cout << " val " << (in[x + y*ncols])[0] << "   " << (in[x + y*ncols])[1] << endl;
        i = std::complex<Type>((in[x + y*ncols])[0], (in[x + y*ncols])[1]);
        //cout << i << endl;
        }

    };
}
#endif


// input store by row-major order in should be outside initialized
void eigen_matrix_to_complex_array(MatrixXcf imga, fftwf_complex* in)
{
    int nw = imga.cols();
    int nh = imga.rows();
    int offset;
    for (int y = 0; y < nh; y++) // rows
    {
        offset = y*nw;
        for (int x = 0; x < nw; x++) // cols
        {
            //float c = imga(y, x).real();
            in[x + offset][0] = imga(y, x).real();
            in[x + offset][1] = imga(y, x).imag();
        }
    }
}

// input store by row-major order in should be outside initialized
void complex_array_eigen_matrix(Eigen::MatrixXcf& imga, fftwf_complex* in)
{
    int ncols = imga.cols();
    int nrows = imga.rows();
    int offset;
    for (int row = 0; row < nrows; row++) // rows
    {
        offset = row*ncols;
        for (int col = 0; col < ncols; col++) // cols
        {
            //cout << " val " << (in[col + offset])[0] << "   " << (in[col + offset])[1] << endl;
            (imga(row, col)).real((in[col + offset])[0]);
            (imga(row, col)).imag((in[col + offset])[1]);
        }
    }
}

// transform a reduced 2d hermit symmetry array to a full 2d array
void hermite_to_array(fftwf_complex* out, fftwf_complex* ret, int nw, int nh)
{
    int nw2, nh2;
    int conjx, conjy;

    nw2 = nw / 2 + 1;
    nh2 = nh / 2 + 1;

    float invsz = 1.0 /*/ (nw * nh)*/;
    int offset, offsetnw2;
    for (int y = 0; y < nh; y++)
    {
        offset = y * nw;
        offsetnw2 = y * nw2;
        for (int x = 0; x < nw2; x++)
        {
            // the zero col is
            if (x == 0)
            {
                conjx = 0;
                conjy = 2 * (nh2 - 1) - y;
            }

            if (y == 0)
            {
                conjy = 0;
                conjx = 2 * (nw2 - 1) - x;
            }

            if (x != 0 && y != 0)
            {
                conjx = 2 * (nw2 - 1) - x;
                conjy = 2 * (nh2 - 1) - y;
            }

            // first half
            ret[x + offset][0] = out[x + offsetnw2][0] * invsz;
            ret[x + offset][1] = out[x + offsetnw2][1] * invsz;

            // 2nd half
            ret[conjx + conjy * nw][0] = out[x + offsetnw2][0] * invsz;
            ret[conjx + conjy * nw][1] = -out[x + offsetnw2][1] * invsz;
        }
    }
}

// transform a reduced 2d hermit symmetry array to a full 2d array
void hermite_to_array_pc(fftwf_complex* out, fftwf_complex* ret, int nw, int nh)
{
    int nw2, nh2;
    int conjx, conjy;

    nw2 = nw / 2 + 1;
    nh2 = nh / 2 + 1;

    float invsz = 1.0 /*/ (nw * nh)*/;
    int offset, offsetnw2;
    for (int y = 0; y < nh; y++)
    {
        offset = y * nw;
        offsetnw2 = y * nw2;
        for (int x = 0; x < nw2; x++)
        {
            ret[offset + x][0] = out[offsetnw2 + x][0];
            ret[offset + x][1] = out[offsetnw2 + x][1];
        }
    }

    int ofst;
    for (int y = 0; y < nh; y++)
    {
        if (y != 0)
        {
            offset = y * nw;
            ofst = (nh - y) * nw;
        }
        else
        {
            offset = y * nw;
            ofst = y * nw;
        }
        for (int x = nw2; x < nw; x++)
        {
            if (x != 0)
            {
                ret[offset + x][0] = ret[ofst + nw - x][0];
                ret[offset + x][1] = -ret[ofst + nw - x][1];
            }
            else
            {
                ret[offset + x][0] = ret[ofst + nw - x][0];
                ret[offset + x][1] = -ret[ofst + nw - x][1];
            }
        }
    }
}

void hermite_to_array_pc_para(fftwf_complex* out, fftwf_complex* ret, int nw, int nh)
{
    int nw2, nh2;
    int conjx, conjy;

    nw2 = nw / 2 + 1;
    nh2 = nh / 2 + 1;

    float invsz = 1.0 /*/ (nw * nh)*/;
    int offset;
    #pragma omp parallel for
    for (int y = 0; y < nh; y++)
    {
        for (int x = 0; x < nw2; x++)
        {
            ret[y * nw + x][0] = out[y * nw2 + x][0];
            ret[y * nw + x][1] = out[y * nw2 + x][1];
        }
    }

    int ofst;
    for (int y = 0; y < nh; y++)
    {
        if (y != 0)
        {
            offset = y * nw;
            ofst = (nh - y) * nw;
        }
        else
        {
            offset = y * nw;
            ofst = y * nw;
        }
        #pragma omp parallel for
        for (int x = nw2; x < nw; x++)
        {
            if (x != 0)
            {
                ret[offset + x][0] = ret[ofst + nw - x][0];
                ret[offset + x][1] = -ret[ofst + nw - x][1];
            }
            else
            {
                ret[offset + x][0] = ret[ofst + nw - x][0];
                ret[offset + x][1] = -ret[ofst + nw - x][1];
            }
        }
    }
}

// 2D fftshift for complex data
void _2d_fft_shift(fftwf_complex* out, int nw, int nh)
{
    int x, y, newx, newy;
    int nh2, nw2;
    nh2 = nh / 2;
    nw2 = nw / 2;

    int idx1, idx2;
    int offset, offsetnew;
    for (y = 0; y < nh2; y++)
    {
        newy = y + nh2;
        offset = y * nw;
        offsetnew = newy * nw;

        // the top-left block to bottom-right
        for (x = 0; x < nw2; x++)
        {
            newx = x + nw2;

            idx1 = x + offset;
            idx2 = newx + offsetnew;

            std::swap(out[idx1][0], out[idx2][0]);
            std::swap(out[idx1][1], out[idx2][1]);
        }

        // top-right to bottom-left block
        for (x = nw2; x < nw; x++)
        {
            newx = x - nw2;

            idx1 = x + offset;
            idx2 = newx + offsetnew;

            std::swap(out[idx1][0], out[idx2][0]);
            std::swap(out[idx1][1], out[idx2][1]);
        }
    }
}

void compute_moment_stats(VectorXf points, int sz, float& vmu, float& vstd)
{
    vmu = 0;
    vstd = 0;
    float a;
    for (int i = 0; i < sz; i++)
    {
        a = points[i];
        vmu += a;
        vstd += a * a;
    }
    vmu = vmu / sz;
    a = vmu*vmu;

    vstd = fabs(vstd / sz - a);
    vstd = sqrt(vstd);
}

// extracting feature by a sliding windows from Gabor response
void slide_windows_features_moments(VectorXf & vecFea, float** pGabMag, int nh, int nw, int nFB, int xblocks, int yblocks)
{
    // initial function pointer for feature extraction
    int fSZ = 2;
    float* pMag;
    int sz = nh * nw;

    // at least 2 blocks in x-orientations and at most 10 blocks
    int xstep, ystep;
    xstep = nw / xblocks;
    ystep = nh / yblocks;

    VectorXf points;
    points.resize(xstep * ystep);

    int x, y, x_begin, x_end, y_begin, y_end;

    //vecFea.clear();
    int nFeaSZ = fSZ * xblocks * yblocks * nFB;
    vecFea.resize(nFeaSZ);

    // k filter banks
    int cnt = 0, offset;
    for (int k = 0; k < nFB; k++) // for each filter-bank
    {
        pMag = pGabMag[k];
        y_end = 0;
        for (int j = 0; j < yblocks; j++) // rows
        {
            y_begin = y_end;
            if (j == 0)
                y_begin = 0;

            y_end = y_begin + ystep;
            if (j == yblocks - 1)
                y_end = nh;

            x_end = 0;
            for (int i = 0; i < xblocks; i++) // cols
            {
                x_begin = x_end;
                if (i == 0)
                    x_begin = 0;

                x_end = x_begin + xstep;
                if (i == xblocks - 1)
                    x_end = nw;

                // push data into an array and calculate the statistics
                int block_sz = (y_end - y_begin) * (x_end - x_begin);
                int ptcnt = 0;
                points.resize(block_sz);
                for (y = y_begin; y < y_end; y++) // rows
                {
                    offset = y * nw;
                    for (x = x_begin; x < x_end; x++) // cols
                    {
                        points[ptcnt] = pMag[x + offset];
                        ptcnt++;
                    }
                }

                // calculate block features
                compute_moment_stats(points, ptcnt, vecFea[cnt], vecFea[cnt + 1]);
                cnt += fSZ;
            } // end i
        } // end j
    } // end k
}

void slide_windows_features_local_maxima(VectorXf & vecFea, float** pGabMag, int nh, int nw, int nFB, int xblocks, int yblocks)
{
    // initial function pointer for feature extraction
    int fSZ = 2;
    float* pMag;
    int sz = nh * nw;

    // at least 2 blocks in x-orientations and at most 10 blocks
    int xstep, ystep;
    xstep = nw / xblocks;
    ystep = nh / yblocks;

    VectorXf points;
    points.resize(xstep * ystep);

    int x, y, x_begin, x_end, y_begin, y_end;

    //vecFea.clear();
    int nFeaSZ = fSZ * xblocks * yblocks * nFB;
    vecFea.resize(nFeaSZ);

    // k filter banks
    int cnt = 0, offset;
    for (int k = 0; k < nFB; k++) // for each filter-bank
    {
        pMag = pGabMag[k];
        y_end = 0;
        for (int j = 0; j < yblocks; j++) // rows
        {
            y_begin = y_end;
            if (j == 0)
                y_begin = 0;

            y_end = y_begin + ystep;
            if (j == yblocks - 1)
                y_end = nh;

            x_end = 0;
            for (int i = 0; i < xblocks; i++) // cols
            {
                x_begin = x_end;
                if (i == 0)
                    x_begin = 0;

                x_end = x_begin + xstep;
                if (i == xblocks - 1)
                    x_end = nw;

                // push data into an array and calculate the statistics
                int block_sz = (y_end - y_begin) * (x_end - x_begin);
                int ptcnt = 0;
                points.resize(block_sz);
                for (y = y_begin; y < y_end; y++) // rows
                {
                    offset = y * nw;
                    for (x = x_begin; x < x_end; x++) // cols
                    {
                        points[ptcnt] = pMag[x + offset];
                        ptcnt++;
                    }
                }

                // calculate block features
                compute_moment_stats(points, ptcnt, vecFea[cnt], vecFea[cnt + 1]);
                cnt += fSZ;
            } // end i
        } // end j
    } // end k
}

std::vector<list_vector> slide_windows_features_local_maxima1(std::vector<MatrixXf> pGabMag, int nh, int nw, int nFB, int xblocks, int yblocks, int wind)
{
    // initial function pointer for feature extraction
    int fSZ = 2;
    MatrixXf pMag;
    int sz = nh * nw;

    // k filter banks
    int cnt = 0, offset;
    std::vector<list_vector> max_locaux(nFB);

    for (int k = 0; k < nFB; k++) // for each filter-bank
    {
        cout << "filter " << k << endl;
        list_vector list_temp;
        list_vector interest_points;
        list_vector interest_pointsr;
        pMag = pGabMag[k];
        int nrows = pMag.rows(), ncols = pMag.cols();
        for (int row = 0; row < nrows; row++) // rows
        {
            for (int col = 0; col < ncols; col++) // cols
            {
                float max = pMag(row, col);
                if (max == 0)
                    continue;
                for (int sub_row = -wind; sub_row <= wind; sub_row++)
                {
                    for (int sub_col = -wind; sub_col <= wind; sub_col++)
                    {
                        if ((sub_row + row >= 0 && sub_row + row < nrows) && (sub_col + col >= 0 && sub_col + col < ncols))
                        {
                            if (pMag(row + sub_row, col + sub_col) > max)
                            {
                                max = pMag(row + sub_row, col + sub_col);
                                sub_row = sub_col = wind + 1;
                            }
                        }
                    }
                }
                if (max == pMag(row, col))
                {
                    //cout << "new point " << max << "  " << col << "  " << row << endl;
                    list_temp.push_back({ max,(float)col,(float)row });
                }
            } // end i
        } // end j
        list_temp.sort([&](std::vector<float>& a, std::vector<float>& b) {return a[0] > b[0]; });
        cout << "list_temp " << list_temp.size() << endl;
        int cp_key_point = 0;
        float max_of;
        for (const auto& x : list_temp)
        {
            if (cp_key_point > 10)
                break;
            std::vector<float> coord = { x[1],x[2] };
            int found = 0;
            for (const auto &it : interest_points)
            {
                std::vector<float> val = it;
                if (fabs(val[1] - coord[0]) < 10 && fabs(val[2] - coord[1]) < 10)
                {
                    found = 1;
                    break;
                }
            }
            if (found == 0)
            {
                //cout << "voila " << endl;
                std::vector<float> cord(3);
                cord[0] = x[0];
                cord[1] = x[1];
                cord[2] = x[2];
                interest_points.push_back(cord);
                //cout << "cp_key_point " << cp_key_point << endl;
                cp_key_point++;
                //interest_points.push_back(cord);
            }
        }
        max_locaux[k] = interest_points;
    } // end k
    return max_locaux;
}

void magnitude_to_image(VectorXf pMag, cv::Mat img)
{
    int nw = img.cols;
    int nh = img.rows;

    float val_max = -FLT_MAX, val_min = FLT_MAX;
    for (int i = 0; i < nw * nh; i++)
    {
        if (pMag[i] > val_max)
            val_max = pMag[i];

        if (pMag[i] < val_min)
            val_min = pMag[i];
    }

    float denorm = 0;
    if (fabs(val_min - val_max) > FLT_EPSILON)
        denorm = 255.0 / (val_max - val_min);
    printf("min = %g, max = %g\n", val_max, val_min);

    int val = 0;
    for (int y = 0; y < nh; y++)
    {
        for (int x = 0; x < nw; x++)
        {
            val = (pMag[y*nw + x] - val_min) * denorm;
            if (val < 0)
                val = 0;
            if (val > 255)
                val = 255;
            img.at<uchar>(y, x) = val;
        }
    }
}
}
