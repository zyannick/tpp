#pragma once


#include <algorithm>
#include <queue>
#include <vector>
#include <iostream>
#include <assert.h>
#include <map>
#include <utility>
#include <string>
#include <complex>
#include <tuple>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;


namespace tpp {

typedef Vector2i pixel_type;
typedef std::multimap< int, pixel_type  > graph_type;
typedef std::priority_queue<pixel_type>  queue_of_pixels;




struct meyer_watershed
{
    inline meyer_watershed();
    inline void process_watershed_algo(MatrixXf &img, int connectivity );
    inline void populate_graph_from_array(const MatrixXf i_image, graph_type  &o_graph);
    inline std::vector< pixel_type > get_neighbors_list(const MatrixXf &input_image, pixel_type input_pixel, int neighborhood_size );

    graph_type image_graph;
    std::queue< pixel_type > fifo;
    MatrixXf lab_w;
    MatrixXf markers;

};

}


#include "watershed_meyer.hpp"
