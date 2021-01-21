#pragma once

#define NOMINMAX

#include <Eigen/Core>
#include <limits>
#include  <iostream>

using namespace Eigen;



namespace tpp {

#define infinie 1E20

using namespace std;


float my_square(float d)
{
    return d*d;
}

VectorXf dt(VectorXf f,int n)
{
    VectorXf d = VectorXf::Zero(n);
    VectorXi v = VectorXi::Zero(n);
    VectorXf z = VectorXf::Zero(n+1);


    int k = 0;


    v[0] = 0;
    z[0] = -infinie;
    z[1] = infinie;

    for(int i = 0 ; i< n ; i++)
    {
        //cout << "f " << f[i] << endl;
    }

    for(int q = 1 ; q <= n-1 ; q++)
    {
        float s  = ( (f[q]+my_square(q)) - (f[v[k]]+my_square(v[k])))/(2*q-2*v[k]);
        //cout << (f[q]+my_square(q)) << "   "  << (f[v[k]]+my_square(v[k])) << "  " << (2*q-2*v[k]) << endl;
        //cout << "s "  << s << " q " << q << " square q " << my_square(q) << endl;
        while (s <= z[k]) {
            k--;
            s  = ((f[q]+my_square(q))-(f[v[k]]+my_square(v[k])))/(2*q-2*v[k]);
        }
        k++;
        v[k] = q;
        z[k] = s;
        z[k+1] = infinie;
    }

    k = 0;

    for (int q = 0; q <= n-1; q++) {
        while (z[k+1] < q)
        {
            k++;
        }
        d[q] = my_square(q-v[k]) + f[v[k]];
    }
    return d;
}

MatrixXf dt(MatrixXf img)
{
    cout << "distance transform" << endl;

    int nrows = img.rows(), ncols = img.cols();
    Eigen::VectorXf f = Eigen::VectorXf((std::max)(nrows,ncols));
    MatrixXf out_img = MatrixXf::Zero(nrows, ncols);

    std::cout << "height "  << nrows << " width " << ncols << std::endl;

    // transform along columns
    for(int col = 0 ; col < ncols ; col++)
    {
        for(int row = 0 ; row < nrows ; row++)
        {
            f(row) = img(row,col);

        }
        VectorXf d = dt(f,nrows);
        for(int row =0 ; row < nrows ; row++)
        {
            img(row,col) = d(row);


        }
    }

    //cout << endl << endl << endl;

    // transform along rows
    //f = VectorXf((std::max)(nrows,ncols));
    for(int row = 0 ; row < nrows ; row ++)
    {
        for(int col = 0 ; col < ncols ; col ++)
        {
            f(col) = img(row,col);

        }
        VectorXf d = dt(f,ncols);
        for(int col = 0 ; col < ncols ; col ++)
        {
            img(row,col) = d(col);
        }
    }


    out_img = img;

    return out_img;
}

MatrixXf efficient_euclidian_distance_transform(MatrixXf img)
{
    unsigned char on = 1;
    int nb_1 = 0;
    int nb_0 = 0;
    int nrows = img.rows(), ncols = img.cols();
    MatrixXf temp_img = MatrixXf::Zero(nrows, ncols);

    for (int row = 0; row < nrows; row++) {
        for (int col = 0; col < ncols; col++) {
            if (img(row,col) == 1)
            {
                nb_1++;
                //cout << "img(row,col) " << img(row,col) << endl;
                temp_img( row, col) = 0;
            }
            else
            {
                nb_0++;
                //cout << "img(row,col) " << img(row,col) << endl;
                temp_img( row, col) = infinie;
            }
        }
    }

    //std::cout << "nb 0 " << nb_0 << " nb 1 " << nb_1 << std::endl;

    MatrixXf res = dt(temp_img);


    for (int row = 0; row < nrows; row++) {
        for (int col = 0; col < ncols; col++) {
            res(row,col) = sqrt(res(row,col));
            //cout << "val " << res(row,col) << endl;
        }
    }


    return(res);
}

MatrixXf euclidian_distance_transform(MatrixXf img,int wr,int wc)
{
    int mid_row = floor(wr/2), mid_col = floor(wc/2);
    int nrows = img.rows(), ncols = img.cols();
    float rho = sqrt(nrows*nrows + ncols*ncols);
    MatrixXf out_img = MatrixXf::Zero(nrows, ncols);
    for(int row = 0 ; row < nrows; row ++)
    {
        for(int col = 0; col < ncols ; col ++)
        {
            cout << "row " << row << " col " << col << endl;
            if(img(row,col)==1)
                out_img(row,col) = 0;
            else
            {
                float dist = rho;
                for(int sub_row = row - mid_row ; sub_row <= row + mid_row ; sub_row++)
                {
                    for(int sub_col = col - mid_col ; sub_col <= col + mid_col ; sub_col++ )
                    {
                        //cout << "sub_row " << sub_row << " sub_col " << sub_col << endl;
                        /*if(sub_row == 1 && sub_col ==1)
                        {
                            cout << "img " << img(sub_row,sub_col) << endl;
                        }*/
                        if(sub_row >= 0 && sub_row < nrows && sub_col >= 0 && sub_col < ncols )
                        {
                            if(!(sub_row == row && sub_col == col) && img(sub_row,sub_col) == 1)
                            {
                                float d = sqrt( my_square(row - sub_row) + my_square(col - sub_col) );
                                cout << "d " << d << endl;
                                dist = (std::min)(dist,d);
                            }
                        }
                    }
                }
                if(dist == rho)
                {
                    out_img(row,col) = 0;
                }
                else
                {
                    out_img(row,col) = dist;
                }
            }
            // cout << endl << endl << endl << endl;
        }
    }
    return out_img;
}

void l1_distance_transform()
{

}

void city_block_distance_transform()
{

}

void chessboard_distance_transform()
{

}

}
