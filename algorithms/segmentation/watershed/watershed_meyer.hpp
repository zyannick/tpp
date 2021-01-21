#pragma once
#include "watershed_meyer.hh"
#include "algorithms/miscellaneous.hh"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"

using namespace  cv;

namespace tpp {


static int nb_times = 0;

bool populate_marker_liste( Matrix<Vector3i, Dynamic, Dynamic> &mat_colored,std::vector<Vector2i> &marker, int row, int col, MatrixXf &mat , Vector3i color_value , size_t i )
{
    int ncols = mat.cols();
    int nrows = mat.rows();
    int nb = 8;

    bool continue_recursion = true;

    mat_colored(row, col) = color_value;

    //cout << row << "   " <<  col << "   " << marker.size() << "  " << nrows << "  " <<  ncols << endl;

    /*for(size_t i = 0 ; i < marker.size() ; i++)
    {
        cout << marker[i](0,0) << "  " << marker[i](1,0) << endl;
    }*/


    /*if(nb_times > 250)
    {
        cout << "called " << nb_times << " liste numero " << i << "   row " <<  row << "  col " <<  col << "   " << endl;
        return continue_recursion;
    }*/

    nb_times++;

    mat(row, col) = 127;



    for(int c = col - 1; c <= col + 1; c ++ )
    {
        if(c < 0 || c >= ncols)
        {
            nb--;
            continue;
        }
        for(int r = row - 1 ; r <= row + 1; r++ )
        {
            if(r < 0 || r >= nrows)
            {
                nb--;
                continue;
            }
            nb--;
            if(mat(r, c) == 255)
            {
                marker.push_back(Vector2i(r, c));
                populate_marker_liste( mat_colored , marker, r, c, mat, color_value , i);
            }
        }
        /*if(nb_times < 250)
        {
            continue_recursion = false;
            cout << "called false  " << nb_times << " liste numero " << i << endl;
        }*/
    }


    return continue_recursion;
}

bool populate_marker_liste_without_recursion( Matrix<Vector3i, Dynamic, Dynamic> &mat_colored,std::vector<Vector2i> &marker, int row, int col, MatrixXf &mat , Vector3i color_value , int i )
{
    int ncols = mat.cols();
    int nrows = mat.rows();
    int nb = 8;
    mat(row, col) = 255 + i + 1;

    mat_colored(row, col) = color_value;

    //cout << row << "   " <<  col << "   " << marker.size() << "  " << nrows << "  " <<  ncols << endl;

    /*for(size_t i = 0 ; i < marker.size() ; i++)
    {
        cout << marker[i](0,0) << "  " << marker[i](1,0) << endl;
    }*/

    //cout << endl << endl;



    for(int c = col - 1; c <= col + 1; c ++ )
    {
        if(c < 0 || c >= ncols)
        {
            nb--;
            continue;
        }
        for(int r = row - 1 ; r <= row + 1; r++ )
        {
            if(r < 0 || r >= nrows)
            {
                nb--;
                continue;
            }
            nb--;
            if(mat(r, c) == 255)
            {
                mat(r, c) = 255 + i + 1;
                marker.push_back(Vector2i(r, c));
            }
        }
    }
    return true;
}

void get_markers(MatrixXf dist_trans, MatrixXf dist_trans_bin , MatrixXf matrix_edge)
{

    cout << "get_markers" << endl;
    int ncols = dist_trans.cols();
    int nrows = dist_trans.rows();

    //get location of maximum
    MatrixXf::Index maxRow_r, maxCol_r;
    float max_value = dist_trans.maxCoeff(&maxRow_r, &maxCol_r);
    //get location of minimum
    MatrixXf::Index minRow_r, minCol_r;
    float min_value = dist_trans.minCoeff(&minRow_r, &minCol_r);

    MatrixXf matrix_max = MatrixXf::Constant(nrows, ncols, max_value);

    dist_trans = matrix_max - dist_trans;

    Matrix<Vector3i, Dynamic, Dynamic> mat_colored(nrows,ncols);
    Matrix<Vector3i, Dynamic, Dynamic> mat_colored_edge(nrows,ncols);

    //get location of maximum
    max_value = matrix_edge.maxCoeff(&maxRow_r, &maxCol_r);
    //get location of minimum
    min_value = matrix_edge.minCoeff(&minRow_r, &minCol_r);

    float factor = float(255.0) / max_value;

    for(int row = 0 ; row < nrows; row++)
    {
        for(int col = 0; col < ncols; col++)
        {
            mat_colored(row, col) = Vector3i(0,0,0);
            int val = int (factor * matrix_edge(row, col));
            mat_colored_edge(row, col) =  Vector3i(val,val,val);
        }
    }


    typedef std::vector<Vector2i> markers;
    std::vector<markers> list_markers;

    Vector3i color_value;

    size_t i = 0;

    for(int row = 0 ; row < nrows; row++)
    {
        for(int col = 0; col < ncols; col++)
        {
            if(int(dist_trans_bin(row,col))==255)
            {
                markers mark;
                mark.push_back(Vector2i(row, col));
                color_value = generate_color(127*3);
                populate_marker_liste(mat_colored ,mark, row, col, dist_trans_bin, color_value , i);
                list_markers.push_back(mark);
                i++;
            }
        }
    }

    mat_colored_edge = mat_colored_edge + mat_colored;

    Mat result_opencv = eigen_to_mat_template<Vector3i>(mat_colored_edge);

    imwrite("pooopopopopo.png",result_opencv);

    cout << "number of regions " << i + 1 << endl;

    return;
}




}
