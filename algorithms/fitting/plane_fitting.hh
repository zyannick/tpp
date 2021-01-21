#pragma once


#include <Eigen/Core>
#include <iostream>
#include <opencv2/core.hpp>
#include <Eigen/Dense>
#include <algorithm>
#include <random>

#define _USE_MATH_DEFINES
#include <math.h>


using namespace Eigen;
using namespace std;
 

namespace tpp {


std::pair<Vector3d, Vector3d> best_plane_from_points(const std::vector<Vector3d> & c)
{
    // copy coordinates to  matrix in Eigen format
    size_t num_atoms = c.size();
    Eigen::Matrix< Vector3d::Scalar, Eigen::Dynamic, Eigen::Dynamic > coord(3, num_atoms);
    for (size_t i = 0; i < num_atoms; ++i) coord.col(i) = c[i];

    // calculate centroid
    Vector3d centroid(coord.row(0).mean(), coord.row(1).mean(), coord.row(2).mean());

    // subtract centroid
    coord.row(0).array() -= centroid.x();
    coord.row(1).array() -= centroid.y();
    coord.row(2).array() -= centroid.z();

    // we only need the left-singular matrix here
    //  http://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points
    auto svd = coord.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);

    cout << "svd " << svd.matrixU() << endl;
    cout << "svd " << svd.matrixV() << endl;
    Vector3d plane_normal = svd.matrixU().rightCols<1>();
    plane_normal = plane_normal / plane_normal.z();
    double d = - centroid.x() * plane_normal.x() - centroid.y() * plane_normal.y() - centroid.z() * plane_normal.z();
    cout << "La valeur de d "  << d << endl;
    return std::make_pair(centroid, plane_normal);
}

std::pair<Vector3d, Vector3d> best_plane_from_points_svd_mean(const std::vector<Vector3d> & c)
{
    // copy coordinates to  matrix in Eigen format
    size_t num_atoms = c.size();
    VectorXd b = VectorXd::Zero(num_atoms);
    Eigen::Matrix< Vector3d::Scalar, Eigen::Dynamic, Eigen::Dynamic > Az(num_atoms, 3 );
    for (size_t i = 0; i < num_atoms; ++i)
    {
        Az.row(i) = Vector3d(c[i].x(), c[i].y(), c[i].z());
    }

    // calculate centroid
    Vector3d centroid(Az.col(0).mean(), Az.col(1).mean(), Az.col(2).mean());

    // subtract centroid
    Az.col(0).array() -= centroid.x();
    Az.col(1).array() -= centroid.y();
    Az.col(2).array() -= centroid.z();

    Eigen::Matrix< Vector3d::Scalar, Eigen::Dynamic, Eigen::Dynamic > A(num_atoms, 3 );
    for (size_t i = 0; i < num_atoms; ++i)
    {
        A.row(i) = Vector3d(Az.row(i).x(), Az.row(i).y(), 1);
        b(i) = Az.row(i).z();
    }

    Vector3d trtr = A.bdcSvd(ComputeThinU | ComputeThinV).solve(b);

    //trtr = trtr / trtr.norm();

    return std::make_pair(trtr, trtr);
}

Vector3d best_plane_from_points_svd(const std::vector<Vector3d>  c)
{
    // copy coordinates to  matrix in Eigen format
    size_t num_atoms = c.size();
    VectorXd b = VectorXd::Zero(num_atoms);
    Eigen::Matrix< Vector3d::Scalar, Eigen::Dynamic, Eigen::Dynamic > Az(num_atoms, 3 );
    for (size_t i = 0; i < num_atoms; ++i)
    {
        Az.row(i) = Vector3d(c[i].x(), c[i].y(), 1);
        b(i) = c[i][2];
    }

    Vector3d trtr = Az.bdcSvd(ComputeFullU | ComputeFullV).solve(b);

    //trtr = trtr / trtr.norm();

    return trtr;
}


std::pair < Vector3d, Vector3d > best_line_from_points(const std::vector<Vector3d> & c)
{
    // copy coordinates to  matrix in Eigen format
    size_t num_atoms = c.size();
    Eigen::Matrix< Vector3d::Scalar, Eigen::Dynamic, Eigen::Dynamic > centers(num_atoms, 3);
    for (size_t i = 0; i < num_atoms; ++i) centers.row(i) = c[i];

    Vector3d origin = centers.colwise().mean();
    Eigen::MatrixXd centered = centers.rowwise() - origin.transpose();
    Eigen::MatrixXd cov = centered.adjoint() * centered;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(cov);
    Vector3d axis = eig.eigenvectors().col(2).normalized();

    return std::make_pair(origin, axis);
}

struct range_generator {
    range_generator(int init) : start(init) { }

    int operator()() {
        return start++;
    }

    int start;
};

std::vector<double> get_dist_to_plane(std::vector<Vector3d>  list_points_3d, Vector4d abcd)
{
    int N = list_points_3d.size();
    std::vector<double> dists(N, 0.0);
    for(int i = 0; i < N; i++)
    {
        Vector3d temp = list_points_3d[i];
        dists[i] = fabs(temp(0)*abcd(0) + temp(1)*abcd(1) + temp(2)*abcd(2) + abcd(3)) / abcd.segment(0,3).norm();
        //cout << "vraiment bizarre " << temp.transpose() << "      " <<  fabs(temp(0)*abcd(0) + temp(1)*abcd(1) + temp(2)*abcd(2) + abcd(3))/abcd.segment(0,3).norm() << "    " << abcd.segment(0,3).norm() << endl;
        //cout << "verite " << dists[i]  << endl << endl;;
    }
    return dists;
}


Vector4d ransac_plane_fitting(std::vector<Vector3d>  coords, int nb_iterations = 1000000,int cp = 5, double inlier_thresh=100.0)
{
    std::vector<Vector3d> max_inlier_list;
    int max_inlier_num = -1;

    int N = coords.size();

    //cout << "taille des points " << N << endl;

    assert( N > 3);

    std::vector<int> indices(N, 0);

    generate(begin(indices), end(indices), range_generator(0));

    for( int iteration = 0 ; iteration < nb_iterations ; iteration++)
    {

        //cout << "ransac iteration " << i << " et taille " << max_inlier_num  <<  endl;
        std::vector<Vector3d> taken(cp);

        std::random_device rd;
        std::mt19937 g(rd());

        std::shuffle(indices.begin(), indices.end(), g);

        for(int it = 0 ; it < cp ; it++)
        {
            taken[it] = coords[indices[it]];
            //cout << "coordonnee " << indices[it] << endl;
        }

        //C[0]*X + C[1]*Y + C[2] - Z = 0
        Vector3d C = best_plane_from_points_svd(taken);


        //ax + by + cz +d = 0
        Vector4d abcd(C.x(), C.y(), -1, C.z());

        std::vector<double> distances = get_dist_to_plane( coords , abcd );

        std::vector<Vector3d> temp_inlier_list;

        for( int j = 0 ; j < N ; j++)
        {
            if( distances[j] <  inlier_thresh )
            {
                //cout << "distance " << distances[j] << endl ;
                temp_inlier_list.push_back(coords[j]);
            }
        }


        if( int(temp_inlier_list.size()) >  max_inlier_num)
        {
            //cout << "youpi on a des inliers " << endl;
            max_inlier_list = temp_inlier_list;
            max_inlier_num = int(temp_inlier_list.size());
        }

        //cout << "plane numero " << iteration << "   " << C[0] << "*X "  <<  C[1] << "*Y " << -1 << "*Z + " << C[2] << " = 0     number of inliers " << temp_inlier_list.size() << endl;

        //cout << "nouvelle taille " << temp_inlier_list.size() << "  " << max_inlier_num << endl;
    }

    cout << "la taille finale " << max_inlier_list.size() << endl;

    //C[0]*X + C[1]*Y + C[2] - Z = 0
    Vector3d C_final = best_plane_from_points_svd(max_inlier_list);
    //ax + by + cz +d = 0
    Vector4d abcd_final(C_final.x(), C_final.y(), -1, C_final.z());

    return abcd_final;

}

Vector4d ransac_plane_fitting_ground_wall(std::vector<Vector3d>  coords, Vector4d abcd_wall, int nb_iterations = 1000000,int cp = 5,
                                          double inlier_thresh=25.0, double dot_cosine_product_threshold = 0.0001)
{

    Vector3d normal_vector_wall = abcd_wall.segment(0,3);

    std::vector<Vector3d> max_inlier_list;

    double min_cosine_dot_product = 10000;
    int max_inlier_num = -1;

    int N = coords.size();

    //cout << "taille des points " << N << endl;

    assert( N > 3);

    std::vector<int> indices(N, 0);

    generate(begin(indices), end(indices), range_generator(0));

    for( int iteration = 0 ; iteration < nb_iterations ; iteration++)
    {

        srand(time(NULL));
        cp = (rand() % 5) + 3;

        //cout << "ransac iteration " << i << " et taille " << max_inlier_num  <<  endl;
        std::vector<Vector3d> taken(cp);

        std::random_device rd;
        std::mt19937 g(rd());

        std::shuffle(indices.begin(), indices.end(), g);

        for(int it = 0 ; it < cp ; it++)
        {
            taken[it] = coords[indices[it]];
        }
        //C[0]*X + C[1]*Y + C[2] - Z = 0
        Vector3d C = best_plane_from_points_svd(taken);
        //ax + by + cz +d = 0
        Vector4d abcd(C.x(), C.y(), -1, C.z());
        std::vector<double> distances = get_dist_to_plane( coords , abcd );
        std::vector<Vector3d> temp_inlier_list;

        Vector3d normal_vector_ground(C.x(), C.y() , -1 );

        double result_of_dot_product = normal_vector_ground.dot(normal_vector_wall);
        double temp_cosine_angle_between_planes = result_of_dot_product / (normal_vector_ground.norm() * normal_vector_wall.norm() );
        temp_cosine_angle_between_planes = fabs(temp_cosine_angle_between_planes);
        //result_of_dot_product = result_of_dot_product.cwiseAbs();

        //cout << "cross product value before testing  " << result_of_dot_product(0) << "   " << result_of_dot_product(1) << "   " << result_of_dot_product(2)  << endl;

        //cout << "cross product value before testing  " << result_of_dot_product  << endl;

        if( temp_cosine_angle_between_planes < dot_cosine_product_threshold  )
        {


            if(min_cosine_dot_product > temp_cosine_angle_between_planes)
            {

                cout << "cross product cosine value " << temp_cosine_angle_between_planes << "     " << acos(temp_cosine_angle_between_planes) << "    "  << taken.size() << endl;

                cout << "Orthogonal " << endl;

                min_cosine_dot_product = temp_cosine_angle_between_planes;

                max_inlier_list = taken;
                max_inlier_num = int(taken.size());

                for( int j = 0 ; j < N ; j++)
                {
                    if( distances[j] <  inlier_thresh )
                    {
                        cout << "distance " << distances[j] << endl ;
                        temp_inlier_list.push_back(coords[j]);
                    }
                }

                if( int(temp_inlier_list.size()) >  max_inlier_num)
                {
                    //cout << "youpi on a des inliers " << endl;
                    max_inlier_list = temp_inlier_list;
                    max_inlier_num = int(temp_inlier_list.size());
                }/**/
            }
        }

    }

    cout << "la taille finale " << max_inlier_list.size()  << "  et le dot produtc donne " << min_cosine_dot_product << endl;

    assert( max_inlier_list.size() > 3  && "Unable to find a plane with these thresholds");

    //C[0]*X + C[1]*Y + C[2] - Z = 0
    Vector3d C_final = best_plane_from_points_svd(max_inlier_list);
    //ax + by + cz +d = 0
    Vector4d abcd_final(C_final.x(), C_final.y(), -1, C_final.z());

    return abcd_final;

}


Vector4d ransac_plane_fitting_ground_n_walls(std::vector<Vector3d>  coords, std::vector<Vector4d> abcd_wall, int nb_iterations = 1000000,int cp = 5,
                                          double inlier_thresh=25.0, double dot_cosine_product_threshold = 0.1)
{
    size_t nb_walls = abcd_wall.size();

    std::vector<Vector3d> normal_vector_wall(abcd_wall.size());

    for(size_t index_wall = 0; index_wall < nb_walls ; index_wall++)
           normal_vector_wall[index_wall] = abcd_wall[index_wall].segment(0,3);

    std::vector<double> result_of_dot_product_wall_per_ground(nb_walls);
    std::vector<double> temp_cosine_angle_between_ground_and_walls(nb_walls);
    std::vector<double> min_cosine_dot_product_walls(nb_walls, 1000);

    std::vector<Vector3d> max_inlier_list;

    double min_cosine_dot_product = 10000;
    int max_inlier_num = -1;

    int N = coords.size();

    //cout << "taille des points " << N << endl;

    assert( N > 3);

    std::vector<int> indices(N, 0);

    generate(begin(indices), end(indices), range_generator(0));

    for( int iteration = 0 ; iteration < nb_iterations ; iteration++)
    {

        srand(time(NULL));
        cp = (rand() % 50) + 3;

        //cout << "ransac iteration " << i << " et taille " << max_inlier_num  <<  endl;
        std::vector<Vector3d> taken(cp);

        std::random_device rd;
        std::mt19937 g(rd());

        std::shuffle(indices.begin(), indices.end(), g);

        for(int it = 0 ; it < cp ; it++)
        {
            taken[it] = coords[indices[it]];
        }
        //C[0]*X + C[1]*Y + C[2] - Z = 0
        Vector3d C = best_plane_from_points_svd(taken);
        //ax + by + cz +d = 0
        Vector4d abcd(C.x(), C.y(), -1, C.z());
        std::vector<double> distances = get_dist_to_plane( coords , abcd );
        std::vector<Vector3d> temp_inlier_list;

        Vector3d normal_vector_ground(C.x(), C.y() , -1 );

        size_t nb_walls_almost_normal_to_ground = 0;
        size_t nb_walls_more_normal_than_previous_iteration = 0;

        for(size_t index_wall = 0 ; index_wall < nb_walls ; index_wall++)
        {
            result_of_dot_product_wall_per_ground[index_wall] = normal_vector_ground.dot(normal_vector_wall[index_wall]);
            temp_cosine_angle_between_ground_and_walls[index_wall] = result_of_dot_product_wall_per_ground[index_wall] / (normal_vector_ground.norm() * normal_vector_wall[index_wall].norm() );
            temp_cosine_angle_between_ground_and_walls[index_wall] = fabs(temp_cosine_angle_between_ground_and_walls[index_wall]);

            //cout << "cosinus plan " << index_wall << "  " << temp_cosine_angle_between_ground_and_walls[index_wall] << endl;

            if(temp_cosine_angle_between_ground_and_walls[index_wall] < dot_cosine_product_threshold)
            {
                nb_walls_almost_normal_to_ground++;
            }

            if(temp_cosine_angle_between_ground_and_walls[index_wall] < min_cosine_dot_product_walls[index_wall])
            {
                nb_walls_more_normal_than_previous_iteration++;
            }
        }



        if( nb_walls_almost_normal_to_ground ==  nb_walls )
        {

            if(nb_walls_more_normal_than_previous_iteration == nb_walls)
            {
                min_cosine_dot_product_walls = temp_cosine_angle_between_ground_and_walls;

                for(size_t index_wall = 0 ; index_wall < nb_walls ; index_wall++)
                {
                    //cout << "Wall number " << index_wall  << "  dot product cosine value " << min_cosine_dot_product_walls[index_wall] << "     " << acos(min_cosine_dot_product_walls[index_wall]) << "    "  << taken.size() << endl;
                }

                //cout << "Orthogonal " << endl;


                max_inlier_list = taken;
                max_inlier_num = int(taken.size());

                /*for( int j = 0 ; j < N ; j++)
                {
                    if( distances[j] <  inlier_thresh )
                    {
                        cout << "distance " << distances[j] << endl ;
                        temp_inlier_list.push_back(coords[j]);
                    }
                }

                if( int(temp_inlier_list.size()) >  max_inlier_num)
                {
                    //cout << "youpi on a des inliers " << endl;
                    max_inlier_list = temp_inlier_list;
                    max_inlier_num = int(temp_inlier_list.size());
                }/**/
            }
        }

    }

    cout << "la taille finale " << max_inlier_list.size()  << "  et le dot produtc donne " << min_cosine_dot_product << endl;

    assert( max_inlier_list.size() > 3  && "Unable to find a plane with these thresholds");

    //C[0]*X + C[1]*Y + C[2] - Z = 0
    Vector3d C_final = best_plane_from_points_svd(max_inlier_list);
    //ax + by + cz +d = 0
    Vector4d abcd_final(C_final.x(), C_final.y(), -1, C_final.z());

    return abcd_final;

}


}
