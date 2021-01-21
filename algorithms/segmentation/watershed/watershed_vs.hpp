#pragma once



#include "watershed_vs.hh"

using namespace std;

namespace tpp {

VectorXf convert_mat_to_vect(MatrixXf mat)
{
    VectorXf vect = VectorXf(mat.cols() * mat.rows());
    int i = 0;
    for(int col = 0 ; col < mat.cols() ; col ++ )
    {
        for(int row = 0 ; col < mat.rows() ; row ++ )
        {
            vect(i) = mat(row,col);
            i++;
        }
    }
    return vect;
}



void vs_watershed::populate_graph_from_array(  const MatrixXf i_image, graph_type  &o_graph)
{

    if(o_graph.empty())
    {
        for(int row=0; row < i_image.rows(); ++row)
        {
            for(int col=0; col < i_image.cols(); ++col)
            {
                int p_val = int(i_image(row,col));
                std::pair<int, pixel_type > val(p_val, pixel_type(row,col));
                o_graph.insert(val);
            }
        }
    }
    else
    {
        //std::cout<<"WARNING: multimap is not empty!"<<std::endl;
        //std::cout<<"         it will be cleaned before populating"<<std::endl;
        //o_graph.clear();
        //populate_graph_from_array(i_image,o_graph );
    }
}



std::vector< pixel_type > vs_watershed::get_neighbors_list(const MatrixXf &input_image, pixel_type input_pixel, int neighborhood_size )
{
    std::vector< pixel_type > neighbors_list;

    size_t nb_row = input_image.rows();
    size_t nb_col = input_image.cols();
    size_t idx_row = input_pixel.y();
    size_t idx_col = input_pixel.x();
    size_t new_idx_row = 0, new_idx_col=0;

    //neighbors_list.push_back( pixel_type(input_pixel.first,input_pixel.second)) ;

    if(neighborhood_size == 4)
    {
        //left
        new_idx_col = idx_col - 1;
        new_idx_row =  idx_row;
        if(idx_col > 0)
        {
            neighbors_list.push_back(pixel_type (new_idx_row, new_idx_col));
        }
        //right
        new_idx_col = idx_col + 1;
        new_idx_row =  idx_row;
        if(new_idx_col < nb_col)
        {
            neighbors_list.push_back(pixel_type (new_idx_row, new_idx_col));
        }
        //top
        new_idx_row =  idx_row + 1;
        new_idx_col = idx_col;
        if(new_idx_row < nb_row)
        {
            neighbors_list.push_back(pixel_type (new_idx_row, new_idx_col));
        }
        //bottom
        new_idx_row =  idx_row - 1;
        new_idx_col = idx_col;
        if( idx_row  > 0)
        {
            neighbors_list.push_back(pixel_type (new_idx_row, new_idx_col));
        }

    }
    else if(neighborhood_size == 8)
    {

        //left
        new_idx_col = idx_col - 1;
        new_idx_row =  idx_row;
        if( idx_col > 0)
        {
            neighbors_list.push_back(pixel_type (new_idx_row, new_idx_col));
        }
        //right
        new_idx_col = idx_col + 1;
        new_idx_row =  idx_row;
        if(new_idx_col < nb_col)
        {
            neighbors_list.push_back(pixel_type (new_idx_row, new_idx_col));
        }
        //top
        new_idx_row =  idx_row - 1;
        new_idx_col = idx_col;
        if( idx_row > 0)
        {
            neighbors_list.push_back(pixel_type (new_idx_row, new_idx_col));
        }
        //bottom
        new_idx_row =  idx_row + 1;
        new_idx_col = idx_col;
        if(new_idx_row  < nb_row)
        {
            neighbors_list.push_back(pixel_type (new_idx_row, new_idx_col));
        }

        // top left
        new_idx_col = idx_col - 1;
        new_idx_row =  idx_row - 1;
        if(new_idx_col >= 0 && idx_row > 0)
        {
            neighbors_list.push_back(pixel_type (new_idx_row, new_idx_col));
        }


        // top right
        new_idx_col = idx_col + 1;
        new_idx_row =  idx_row - 1;
        if(new_idx_col < nb_col && idx_row > 0)
        {
            neighbors_list.push_back(pixel_type (new_idx_row, new_idx_col));
        }


        // bottom left
        new_idx_col = idx_col - 1;
        new_idx_row =  idx_row + 1;
        if( idx_col > 0 && new_idx_row < nb_row)
        {
            neighbors_list.push_back(pixel_type (new_idx_row, new_idx_col));
        }

        // bottom right
        new_idx_col = idx_col + 1;
        new_idx_row =  idx_row + 1;
        if(new_idx_col < nb_col && new_idx_row < nb_row)
        {
            neighbors_list.push_back(pixel_type (new_idx_row, new_idx_col));
        }

    }
    else
    {
        std::cout<<"BAOWW !! MAUVAIS VOISINAGE " << std::endl;
    }

    return neighbors_list; // vecteur de max 8 elt on s'autorise la copie
}

inline vs_watershed::vs_watershed()
{

}



inline
void vs_watershed::process_watershed_algo(MatrixXf &img, int connectivity)
{
    int init_tag = -1;
    int mask_tag = -2;
    int wshed_tag = 0;
    pixel_type fictitious = pixel_type(-1,-1);

    int curlab  = 0;
    assert( this->fifo.empty()); // ligne 9 algo

    lab_w = img; //ligne 10 to 12
    MatrixXf dist = MatrixXf::Zero(img.rows(), img.cols());  // algo


    this->populate_graph_from_array(img, this->image_graph); // ligne 13 algo

    //get location of maximum
    MatrixXf::Index maxRow, maxCol;
    int max_ = lab_w.maxCoeff(&maxRow, &maxCol);
    //get location of minimum
    MatrixXf::Index minRow, minCol;
    int min_ = lab_w.minCoeff(&minRow, &minCol);

    for(int level = min_ ; level <= max_ ; level++)
    {
        std::pair< graph_type::iterator , graph_type::iterator >  pixels_at_level_it;
        pixels_at_level_it = this->image_graph.equal_range(level);


        for( graph_type::iterator map_it = pixels_at_level_it.first; map_it !=  pixels_at_level_it.second; ++map_it)
        {
            int row,col;
            row = (*map_it).second.y();
            col = (*map_it).second.x();
            lab_w(row,col) = mask_tag;
            std::vector< pixel_type > neighbors_list = this->get_neighbors_list(img,  (*map_it).second, connectivity);

            for(std::vector< pixel_type >::iterator it=neighbors_list.begin(); it != neighbors_list.end(); ++it)
            {
                pixel_type current_neighbor = *it; // q dans l'algo
                if(lab_w(current_neighbor.y(),current_neighbor.x()) > 0
                        || lab_w( current_neighbor.y(),current_neighbor.x()) == wshed_tag)
                {
                    dist( (*map_it).second.y(), (*map_it).second.x() ) = 1;
                    this->fifo.push( (*map_it).second);
                    break;
                }
            }
        }

        int curdist = 1;
        this->fifo.push(fictitious);

        while(true)
        {
            pixel_type current_pixel = this->fifo.front(); // p dans l'algo (ligne27)
            this->fifo.pop();
            if( current_pixel.y() == fictitious.y() &&  current_pixel.x() == fictitious.x())
            {
                if(this->fifo.empty())
                {
                    break;
                }
                else
                {
                    this->fifo.push(fictitious);
                    curdist += 1;
                    current_pixel = this->fifo.front();
                    this->fifo.pop();
                }
            } // endif (ligne 35 algo)

            std::vector< pixel_type > cur_neighbors_list = this->get_neighbors_list(img,  current_pixel, connectivity);

            for(std::vector< pixel_type >::iterator it=cur_neighbors_list.begin(); it != cur_neighbors_list.end(); ++it)
            {
                pixel_type current_neighbor = *it;// q ligne 36 dans l'algo
                int row,col;
                row = current_neighbor.y();
                col = current_neighbor.x();
                int cur_col,cur_row;
                cur_row = current_pixel.y();
                cur_col = current_pixel.x();
                if( (dist(row,col) < curdist )
                        && ( lab_w(row,col) > 0
                             || lab_w(row,col) == wshed_tag ) )
                {
                    if( lab_w(row,col) > 0)  // ligne 39
                    {
                        if(lab_w(cur_row,cur_col) == mask_tag || lab_w(cur_row,cur_col) == wshed_tag ) // ligne 40
                        {
                            lab_w(cur_row,cur_col) = lab_w(row,col) ;
                        }
                        else if( lab_w(cur_row,cur_col) != lab_w( cur_row,cur_col) )// lig90 42
                        {
                            lab_w(cur_row,cur_col) = wshed_tag;
                        }
                    }
                    else if (lab_w(cur_row,cur_col) == mask_tag) // ligne 45
                    {
                        lab_w(cur_row,cur_col) = wshed_tag;
                    }// enfdif (ligne 47 algo)
                }
                else if(lab_w(row,col)  == mask_tag && dist(row,col) ==0) //ligne 48
                {
                    dist(row,col)= curdist + 1;
                    this->fifo.push(current_neighbor);
                }// endif (ligne 50 algo)
            } //endfor (ligne 51 algo)

        }//end while


        // detect and process new minima at level h
        for( graph_type::iterator map_it = pixels_at_level_it.first; map_it !=  pixels_at_level_it.second; ++map_it)
        {
            pixel_type current_pixel = (*map_it).second;
            int cur_col = current_pixel.x(),cur_row = current_pixel.y();
            dist( cur_row, cur_col) = 0;
            if(lab_w( cur_row, cur_col)  == mask_tag)
            {
                curlab += 1;
                this->fifo.push(current_pixel);
                lab_w( cur_row, cur_col)  =  curlab;
                while( !this->fifo.empty())
                {
                    pixel_type removed_pix = this->fifo.front();
                    this->fifo.pop();
                    std::vector< pixel_type > cur_neighbors_list = this->get_neighbors_list(img, removed_pix, connectivity);
                    for(std::vector< pixel_type >::iterator it=cur_neighbors_list.begin(); it != cur_neighbors_list.end(); ++it)
                    {
                        pixel_type current_neighbor = *it; // r ligne 61
                        int cur_neigh_row = current_neighbor.y(), cur_neigh_col = current_neighbor.x();
                        if (lab_w(cur_neigh_row,cur_neigh_col) == mask_tag)
                        {
                            this->fifo.push( current_neighbor);
                            lab_w(cur_neigh_row,cur_neigh_col) = curlab;
                        }
                    }
                }
            } // endif (ligne 67 algo)
        } // endfor (ligne 68 algo)
    }

    //loop to mark all watershed point
    for ( graph_type::iterator it = this->image_graph.begin(); it!= this->image_graph.end(); ++it)
    {
        pixel_type current_pixel = (*it).second;
        int cur_col = current_pixel.x(),cur_row = current_pixel.y();
        std::vector< pixel_type > neighbors_list = get_neighbors_list(img, current_pixel, 4 );
        int current_label = lab_w(cur_row,cur_col);
        for (std::vector<pixel_type>::iterator nit=neighbors_list.begin(); nit!=neighbors_list.end(); ++nit)
        {
            pixel_type  current_neighbor = *nit;
            int cur_neigh_row = current_neighbor.y(), cur_neigh_col = current_neighbor.x();
            if(lab_w(cur_neigh_row,cur_neigh_col) != wshed_tag && lab_w(cur_neigh_row,cur_neigh_col)<current_label)
            {
                lab_w(cur_row,cur_col) = wshed_tag;
                break;
            }
        }

    }


}

}
