#pragma once

#include <Eigen/Dense>
#include <Eigen/Core>
#include <list>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "core.hpp"
#ifdef vpp
#include "vpp/vpp.hh"
#endif
#include <chrono>

#include "calibration_and_3d/calibration/stereo_calibration.hh"
#include "calibration_and_3d/recontruction3D/recontruction3D.hh"

#include "lades_similarity.hh"

#include "l2_matching.hh"
#include "pc_results.hh"
#include "phase_correlation.hh"
#include "utils_matching.hh"
#include "save_results.hh"


using namespace Eigen;
using namespace tpp;
 
#ifdef vpp
using namespace vpp;
#endif
using namespace std::chrono;


namespace tpp
{


int compute_A_PC(VectorXf PC,int max_col,int s,VectorXf &vect_p,VectorXf &vect_d,int L_max)
{
    int nrows = PC.rows();
    //cout << "nrows " << nrows << endl;
    MatrixXf::Index maxIndex;
    float max_value = PC.maxCoeff(&maxIndex);
    int a = maxIndex;
    int b;
    if(a>1 && a < PC.rows()-1)
    {
        b = PC(a-1) > PC(a+1) ? a - 1 : a + 1;
    }
    else
    {
        //cout << "Vecteur  " << PC << endl;
        return -1;
    }

    //cout << "a " << a << "  b " << b << endl;

    int ecart = min(nrows - 1 - a, nrows - 1 - b);
    if(ecart > a || ecart > b)
        ecart = min(a,b);
    int L = (ecart%2 == 0 ? ecart : ecart - 1);
    if(L<=0)
    {
        return -1;
    }
    // cout << " L " << L << endl;
    if(L>L_max)
        L = L_max;
    L = L*2;

    vect_p = VectorXf::Zero(L);
    vect_d = VectorXf::Zero(L);

    for(int k = 0 ; k < L/2 ; k++)
    {
        vect_p(k) = a;
        vect_d(k) = k + 1;
    }

    for(int k = L/2 ; k < L ; k++)
    {
        vect_p(k) = b;
        vect_d(k) = k + 1 - L/2;
    }

    //cout << " vect " << vect_p << endl << endl;
    return 1;
}


void u_v_phase_correlation(VectorXf &vect_u,VectorXf &vect_v,VectorXf vect_p, VectorXf vect_d,VectorXf PC,int mwc)
{
    int nrows = vect_p.rows();
    vect_u = VectorXf::Zero(nrows);
    vect_v = VectorXf::Zero(nrows);
    for(int row = 0 ; row < nrows ; row++ )
    {
        int p_idx = vect_p(row);
        int p = p_idx - mwc;
        int d = vect_d(row);
        //cout << "p " << p << " p_idx " << p_idx << "   d " << d << endl;
        vect_u(row) = PC(p_idx-d) + PC(p_idx+d) - 2 * cos(M_PI*d)* PC(p_idx);
        vect_v(row) = 2 * p * cos(d*M_PI) * PC(p_idx) - (p - d) * PC(p_idx-d) - (p+d) * PC(p_idx+d);
        //cout << "U "  << vect_u(row) << "  V " << vect_v(row) << endl;
    }
    //cout << endl << endl << endl;
}

void u_v_phase_correlation_weighted(VectorXf &vect_u,VectorXf &vect_v,VectorXf vect_p, VectorXf vect_d,VectorXf PC,int mwc,int V)
{
    int N = 2*mwc +1;
    float v_on_n = (float)V / float(N);
    //cout << "V/N = " << v_on_n << "  N " << N << " V " << V << endl;
    int nrows = vect_p.rows();
    vect_u = VectorXf::Zero(nrows);
    vect_v = VectorXf::Zero(nrows);
    for(int row = 0 ; row < nrows ; row++ )
    {
        int p_idx = vect_p(row);
        int p = p_idx - mwc;
        int d = vect_d(row);
        //cout << "p " << p << " p_idx " << p_idx << "   d " << d << endl;
        vect_u(row) = PC(p_idx-d) + PC(p_idx+d) - 2 * cos(v_on_n*M_PI*d)* PC(p_idx);
        vect_v(row) = 2 * p * cos(v_on_n*d*M_PI) * PC(p_idx) - (p - d) * PC(p_idx-d) - (p+d) * PC(p_idx+d);
        //cout << "U "  << vect_u(row) << "  V " << vect_v(row) << endl;
    }
    //cout << endl << endl << endl;
}


int resolve_pc_nagashima(MatrixXf left_block, MatrixXf right_block, VectorXf &vect_u, VectorXf &vect_v, int mwc , MatrixXf H , int mode ,int L_max, int U)
{
    MatrixXf phase_co;
    phase_co = phase_correlation(left_block,right_block);
    phase_co = fftshift_matrix<float>(phase_co);

    if(mode==1)
        phase_co = phase_co.cwiseProduct(H);



    MatrixXf::Index max_row_after_shift, max_col_after_shift;

    float m_l_after = phase_co.maxCoeff(&max_row_after_shift, &max_col_after_shift);

    VectorXf vect_p,vect_d;
    VectorXf rec = phase_co.row(max_row_after_shift);

    int r;

    r = compute_A_PC(rec,max_col_after_shift,1,vect_p,vect_d,L_max);

    if(r == -1)
    {
        return 0;
    }

    int L = vect_d.rows();

    int V = 2*U+1;

    if(mode == 0)
        u_v_phase_correlation(vect_u,vect_v, vect_p,  vect_d, rec,mwc);
    else
        u_v_phase_correlation_weighted(vect_u,vect_v, vect_p,  vect_d, rec,mwc,V);

    return 1;

}

std::pair<float, float> resolve_pc_forosh(MatrixXf left_block, MatrixXf right_block, VectorXf &vect_u, VectorXf &vect_v, int mwc , MatrixXf H , int mode ,int L_max, int U)
{
    MatrixXf phase_co;
    phase_co = phase_correlation(left_block,right_block);
    phase_co = fftshift_matrix<float>(phase_co);

    if(mode==1)
        phase_co = phase_co.cwiseProduct(H);


    MatrixXf::Index max_row_after_shift, max_col_after_shift;

    float m_l_after = phase_co.maxCoeff(&max_row_after_shift, &max_col_after_shift);

    VectorXf vect_p,vect_d;
    VectorXf rec = phase_co.row(max_row_after_shift);

    float M_positive = phase_co( max_row_after_shift , max_col_after_shift + 1 ) +  phase_co( max_row_after_shift, max_col_after_shift  );
    float M_negative = phase_co( max_row_after_shift , max_col_after_shift + 1 ) -  phase_co( max_row_after_shift, max_col_after_shift  );

    float N_positive = phase_co( max_row_after_shift + 1, max_col_after_shift  ) +  phase_co( max_row_after_shift, max_col_after_shift  );
    float N_negative = phase_co( max_row_after_shift + 1, max_col_after_shift  ) -  phase_co( max_row_after_shift, max_col_after_shift  );

    float delta_x;
    float t_pos_x = phase_co( max_row_after_shift , max_col_after_shift + 1 )  / M_positive ;
    float t_neg_x = phase_co( max_row_after_shift , max_col_after_shift + 1 )  / M_negative ;

    float delta_y;
    float t_pos_y = phase_co(max_row_after_shift + 1, max_col_after_shift) / N_positive ;
    float t_neg_y = phase_co(max_row_after_shift + 1, max_col_after_shift) / N_negative ;

    //cout << "positive " << t_pos << " negative " << t_neg << endl;

    if(fabs(t_pos_x) < 1)
    {
        delta_x = t_pos_x;
    }
    else if( fabs(t_neg_x) < 1)
    {
        delta_x = t_neg_x;
    }


    if(fabs(t_pos_y) < 1)
    {
        delta_y = t_pos_y;
    }
    else if( fabs(t_neg_y) < 1)
    {
        delta_y = t_neg_y;
    }

    return std::pair(delta_y, delta_x);

}




void phase_correlation_sub_pixel_matching_nagashima(MatrixXf PC_left, MatrixXf PC_right, std::vector<stereo_match> &epipolar_order_constraints ,
                                                    int wind = 3, int with_weitghing = 0, int cp = 0, string s ="", float i_ = 0)
{

    std::string file_name;


    //file_name = std::string("../test_disp/weighted_").append(std::to_string(cp)).append(std::string("_").append(s)).append("_").append(std::to_string(2*wind+1)).append(std::string("_.txt"));

    file_name = std::string("../test_disp/weighted.txt");

    if(cp ==0 && i_ ==0)
    {
        std::experimental::filesystem::remove(file_name);
    }

    cout << "file_name " << file_name << endl;

    ofstream myfile (file_name, std::ios_base::app);

    for(wind = 3; wind <= 20; wind++)
    {
        int mwc = wind;
        int pad = wind;
        MatrixXf img_left ,img_right ,H  ;
        fill_border_zero(PC_left, img_left, pad);
        fill_border_zero(PC_right, img_right, pad);


        H = MatrixXf::Zero(2*mwc+1,2*mwc+1);
        int U1,U2;
        U1 = mwc;
        U2 = mwc;
        if(with_weitghing==1)
            rectangular_low_pass(H,U1,U2,mwc);



        std::vector<stereo_match> epipolar_order_constraints_sub_pix;

        for (auto &smt : epipolar_order_constraints)
        {
            float sim = smt.similarity;

            int mid_x_l = int(smt.first_point.x()) + pad;
            int mid_y_l = int(smt.first_point.y()) + pad;
            int mid_x_r = int(smt.second_point.x()) + pad;
            int mid_y_r = int(smt.second_point.y()) + pad;

            float offset_eps;

            if(mid_x_l >= mwc && mid_y_l >=mwc && mid_x_r >=mwc && mid_y_r >= mwc)
            {
                MatrixXf left_block = img_left.block(mid_y_l - mwc,mid_x_l - mwc, 2*mwc+1, 2*mwc+1 );
                MatrixXf right_block = img_right.block(mid_y_r - mwc,mid_x_r - mwc, 2*mwc+1, 2*mwc+1 );

                int L_max = 2;


                VectorXf vect_u;
                VectorXf vect_v;
                float r1 = resolve_pc_nagashima(left_block,right_block ,vect_u , vect_v ,mwc , H, with_weitghing ,L_max,U1 );
                if(r1 ==0)
                {
                    cout << "r1" << endl;
                    continue;
                }

                VectorXf trtr = vect_u.bdcSvd(ComputeThinU | ComputeThinV).solve(vect_v);



                offset_eps = trtr(0);

                if(fabs(offset_eps)> 1)
                    continue;

                float offset_new = mid_x_r + offset_eps - mid_x_l ;

                if (myfile.is_open()  )
                {
                    myfile << std::fixed << std::setprecision(5) << cp << ";";
                    myfile << std::fixed << std::setprecision(5) << i_ << ";";
                    myfile << std::fixed << std::setprecision(5) << mid_x_r - mid_x_l << ";";
                    myfile << std::fixed << std::setprecision(5) << offset_eps << ";";
                    myfile << std::fixed << std::setprecision(5) << offset_new << ";";
                    myfile << std::fixed << std::setprecision(5) << sim <<  ";";
                    myfile << std::fixed << std::setprecision(5) << vect_u.rows() <<  ";";
                    myfile << std::fixed << std::setprecision(5) << 2*L_max << "\n";
                }

            }

            smt.second_point = Vector2d(mid_y_r, mid_x_r + offset_eps);
            epipolar_order_constraints_sub_pix.push_back(smt);
        }

        epipolar_order_constraints = epipolar_order_constraints_sub_pix;
    }


}


void phase_correlation_sub_pixel_matching_forosh(MatrixXf PC_left, MatrixXf PC_right, std::vector<stereo_match> &epipolar_order_constraints , int wind = 3, int with_weitghing = 0)
{

    int mwc = wind;
    int pad = wind;
    MatrixXf img_left ,img_right ,H  ;
    fill_border_zero(PC_left, img_left, pad);
    fill_border_zero(PC_right, img_right, pad);


    H = MatrixXf::Zero(2*mwc+1,2*mwc+1);
    int U1,U2;
    U1 = mwc;
    U2 = mwc;
    if(with_weitghing==1)
        rectangular_low_pass(H,U1,U2,mwc);

    std::vector<stereo_match> epipolar_order_constraints_sub_pix;

    for (auto smt : epipolar_order_constraints)
    {
        //stereo_match smt = l;
        float sim = smt.similarity;


        int mid_x_l = int(smt.first_point.x()) + pad;
        int mid_y_l = int(smt.first_point.y()) + pad;
        int mid_x_r = int(smt.second_point.x()) + pad;
        int mid_y_r = int(smt.second_point.y()) + pad;

        if(mid_x_l >= mwc && mid_y_l >=mwc && mid_x_r >=mwc && mid_y_r >= mwc)
        {
            MatrixXf left_block = img_left.block(mid_y_l - mwc,mid_x_l - mwc, 2*mwc+1, 2*mwc+1 );
            MatrixXf right_block = img_right.block(mid_y_r - mwc,mid_x_r - mwc, 2*mwc+1, 2*mwc+1 );

            int L_max = 100;

            VectorXf vect_u;
            VectorXf vect_v;
            std::pair<float, float> delta  = resolve_pc_forosh(left_block, right_block ,vect_u , vect_v ,mwc , H, with_weitghing ,L_max,U1 );

            smt.second_point.y() = smt.second_point.y() + delta.first;
            smt.second_point.x() = smt.second_point.x() + delta.second;


        }

        epipolar_order_constraints_sub_pix.push_back(smt);
    }

    epipolar_order_constraints = epipolar_order_constraints_sub_pix;

}




void ensure_left_right_consistency(std::vector<stereo_match> left_right_matches, std::vector<stereo_match> righ_left_matches,
                                   std::vector<stereo_match> &left_right_consitensy_matches, Vector2d err_m)
{
    for (int i = 0; i < left_right_matches.size(); i++)
    {
        stereo_match left_to_right = left_right_matches[i];
        for (int j = 0; j < righ_left_matches.size(); j++)
        {
            stereo_match right_to_left = righ_left_matches[j];
            //cout << left_to_right  << "   "  << right_to_left  << endl;
            if( (left_to_right.first_point - right_to_left.first_point).norm() < 2 &&
                    (left_to_right.second_point - right_to_left.second_point).norm() < 2 &&
                    righ_left_matches[j].taken_left_right_consistency != 1 )
            {
                righ_left_matches[j].taken_left_right_consistency = 1;
                left_right_consitensy_matches.push_back(left_to_right);
            }
        }
    }
}

void ensure_left_right_uniqueness(std::vector<stereo_match> left_right_matches, std::vector<stereo_match> &left_right_uniqueness)
{
    for (auto& x : left_right_matches)
    {
        Vector2d coord;
        coord = x.first_point;
        int found = 0;
        for (auto& it : left_right_uniqueness)
        {
            Vector2d val = it.first_point;
            if (   ( coord - val  ).norm() < 2)
            {
                found = 1;
                break;
            }
        }
        if (found == 0)
        {
            left_right_uniqueness.push_back(x);
        }
    }
}

bool contains_matches(std::vector<stereo_match> list_matches, stereo_match val)
{
    for(auto st : list_matches)
    {
        if ( (st.first_point - val.first_point).norm() == 0 ) return true;
    }
    return false;
}

struct compute_time{
    high_resolution_clock::time_point start = high_resolution_clock::now();
    high_resolution_clock::time_point end;
    string where;

    inline
    compute_time()
    {

    }

    inline
    compute_time(string wh)
    {
        where = wh;
    }

    void end_time()
    {
        end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>( end - start ).count();
        //cout << "la duree " << where << "  " << duration << endl ;
    }
};





template< typename Type>
std::vector<stereo_match> matching_stereo_ordering_constraint(phase_congruency_result<MatrixXf> pcr,
                                                              base_options<Type> stm, int cp = 0,MatrixXf otsu_left = MatrixXf::Zero(0,0),
                                                              MatrixXf otsu_right = MatrixXf::Zero(0,0))
{


    std::vector<stereo_match> left_right_matches;
    std::vector<stereo_match> righ_left_matches;

    std::map<key_type, float> map_sim;

    if (MATCHING_SIMILARITY_METHOD::L2_DISTANCE == stm.dist_method)
    {
        //matching_l2(nrows, ncols, ft_left, ft_right, left_right_matches, im_p_all, eps, wind_row, wind_col);
    }
    else if (MATCHING_SIMILARITY_METHOD::LADES_SIMILARITY == stm.dist_method)
    {
        compute_time t_match(" of matching ");

        //cout << "left to right " << endl;
        matching_similarity(left_right_matches, pcr, stm,  1, map_sim, otsu_left, otsu_right);
        std::sort(left_right_matches.begin(), left_right_matches.end(), decrescendo_similarity());


        //cout << "right to left " << endl;
        matching_similarity(righ_left_matches, phase_congruency_result<MatrixXf>(pcr, true), stm, 2, map_sim, otsu_right, otsu_left);
        std::sort(righ_left_matches.begin(), righ_left_matches.end(), decrescendo_similarity());

        t_match.end_time();

    }
    //cout << "end matching " << endl;

    std::vector<stereo_match> left_right_consitensy_matches;

    int nb_matches = left_right_matches.size();

    //cout << "number of matches " << nb_matches << endl;

    if(nb_matches == 0)
        return left_right_matches;


    Vector2d err_m;
    err_m.x() = 1;
    err_m.y() = 1;



    if(stm.ensure_consistency)
    {
        ensure_left_right_consistency(left_right_matches, righ_left_matches,
                                      left_right_consitensy_matches, err_m);
    }
    else {
        left_right_consitensy_matches = left_right_matches;
    }

    //cout << "consistency  " << left_right_consitensy_matches.size() << endl;


    if(left_right_consitensy_matches.size() == 0)
    {
        return left_right_consitensy_matches;
    }

    std::vector<stereo_match> left_right_uniqueness;

    if(stm.ensur_unique)
    {
        ensure_left_right_uniqueness(left_right_consitensy_matches, left_right_uniqueness);
    }
    else
    {
        left_right_uniqueness = left_right_consitensy_matches;
    }

    if(left_right_uniqueness.size() == 0)
    {
        return left_right_uniqueness;
    }


    //cout << "uniqueness 1 " << left_right_uniqueness.size() << endl;


    std::vector<stereo_match> right_left_uniqueness;

    if(stm.ensur_unique)
    {
        ensure_left_right_uniqueness(left_right_uniqueness, right_left_uniqueness);
    }
    else{
        right_left_uniqueness = left_right_uniqueness;
    }

    if(right_left_uniqueness.size() == 0)
    {
        return right_left_uniqueness;
    }

    //cout << "uniqueness 2 " << right_left_uniqueness.size() << endl;


    Vector2d fv = left_right_uniqueness.front().first_point;
    Vector2d  sv = left_right_uniqueness.front().second_point;


    float moy = sv.x() - fv.x();


    std::vector<stereo_match> epipolar_order_constraints;

    for (auto & value : right_left_uniqueness)
    {
        float d = value.second_point.x() - value.first_point.x();
        if (approx_number(d, moy, 10))
        {
            epipolar_order_constraints.push_back(value);
        }
    }

    //cout << "disparity range " << epipolar_order_constraints.size() << endl;

    //save_stereo_images_part(pcr.pc_first, pcr.pc_second,  stm.eps, left_right_consitensy_matches,cp);

    //phase_correlation_sub_pixel_matching_forosh(PC_left,  PC_right, epipolar_order_constraints );



    if(SUB_PIXEL_MATCHING_PRECISE::PHASE_CORRELATION_FORROSH == stm.sub_pixel)
    {
        //phase_correlation_sub_pixel_matching_forosh(PC_left,  PC_right, epipolar_order_constraints );

    }
    else if(SUB_PIXEL_MATCHING_PRECISE::PHASE_CORRELATION_NAGASHIMA == stm.sub_pixel)
    {
        //phase_correlation_sub_pixel_matching_nagashima(PC_left,  PC_right, epipolar_order_constraints, wind_col, 1, cp, "", offset );
        //int wind = 3, int with_weitghing = 0, int cp = 0, string s ="", float i_ = 0
    }



    //cout << "number of matches at the end-->" << epipolar_order_constraints.size() << endl;

    return epipolar_order_constraints;


}

Mat compute_sobel_gradient( Mat input)
{

    int scale = 1;
    int delta = 0;
    int ddepth = CV_64F;

    Mat grad;

    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;

    /// Gradient X
    Sobel( input, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
    /// Gradient Y
    Sobel( input, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_x, abs_grad_x );
    convertScaleAbs( grad_y, abs_grad_y );

    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );


    return grad;

}



template< typename Type>
void matching_ir_stereo_mat_pair(cv::Mat left_view , cv::Mat right_view, phase_congruency_result<MatrixXf> &pcr,
                                 std::vector<stereo_match> &matchings_stereo, base_options<Type> stm,
                                 int index_img = 0, string w_dir = "", MatrixXf otsu_left = MatrixXf::Zero(0,0),
                                 MatrixXf otsu_right = MatrixXf::Zero(0,0))
{
    int nrows,ncols;


    nrows = left_view.rows;
    ncols = left_view.cols;

    MatrixXf PC_left, ft_left, orient_left;
    float T = -100;

    MatrixXf PC_right, ft_right,  orient_right;


    if(stm.type_extraction == 0)
    {
        if( stm.phase_cong_type == 0 )
        {

        }
        else
        {
            phase_congruency<float> pc_set;

            MatrixXf input_left_image = mat_to_eigen(left_view);

            /*int nrows = input_left_image.rows();
            int ncols = input_left_image.cols();
            for(int row = 0; row < nrows; row++)
            {
                for(int col = 0; col < ncols ; col++)
                {
                    cout << input_left_image(row, col) << endl;
                }
            }

            /*MatrixXf img = MatrixXf::Zero(5,6);
            int nrows = img.rows();
            int ncols = img.cols();
            for(int row = 0; row < nrows; row++)
            {
                for(int col = 0; col < ncols ; col++)
                {
                    img(row, col) = col * cos(col) + row * sin(row);
                }
            }*/
            phase_congruency_output_eigen<float> pc_output_left(nrows, ncols, pc_set);
            phase_congruency_3(input_left_image, pc_output_left);

            //cout << 255 * pc_output_left.PC[0] << endl;



            MatrixXf input_right_image = mat_to_eigen(right_view);
            phase_congruency_output_eigen<float> pc_output_right(nrows, ncols, pc_set);
            phase_congruency_3(input_right_image, pc_output_right);

            create_directory("save_temp");
            create_directory("save_temp/sub_" + w_dir);
            string wts = "save_temp/sub_" + w_dir + "/pc_" + std::to_string(index_img);
            create_directory(wts);


            for(int orient = 0; orient < pc_output_left.PC.size(); orient++)
            {
                //cout << 255 * pc_output_left.PC[orient]  << endl;
                cv::Mat temp = eigen_to_mat(255 * pc_output_left.PC[orient]);
                cv::imwrite(wts + "/pc_left_" + std::to_string(index_img) + "_" + std::to_string(orient) + ".png", temp);
            }

            pcr.set_values(pc_output_left.M, pc_output_right.M,
                           pc_output_left.featType, pc_output_right.featType,
                           pc_output_left.orientation, pc_output_right.orientation);
            //cout << "phase congruency end" << endl;

        }

    }
    else if(stm.type_extraction == 1){



        Mat grad_left = compute_sobel_gradient(left_view);
        Mat grad_right = compute_sobel_gradient(right_view);

        PC_left = mat_to_eigen(grad_left);
        PC_right = mat_to_eigen(grad_right);

        pcr.set_values(PC_left,PC_right);


    }

    bool para = false;
    if(!para)
    {
        auto start = std::chrono::high_resolution_clock::now();
        cout << endl << "matching start " << endl;
        matchings_stereo =  matching_stereo_ordering_constraint(pcr, stm , index_img, otsu_left, otsu_right);/**/
        auto end = std::chrono::high_resolution_clock::now();
        cout << "matching end " << duration_cast<milliseconds>(end - start).count() << endl << endl;
    }
    else {
        cout << "matching start ||" << endl;
        /*image2d<float> new_pc_left(nrows, ncols);
        image2d<float> new_pc_right(nrows, ncols);

        pixel_wise(new_pc_left, new_pc_right, new_pc_left.domain()) | [&] ( auto &l, auto &r,  vint2 coord) {
            int x = coord.x();
            int y = coord.y();
            l = PC_left(y, x);
            r = PC_right(y, x);
        };



        matchings_stereo =  matching_stereo_ordering_constraint(nrows, ncols, 3, 3,
                                                                eps,rect, new_pc_left, new_pc_right,stereo_par,cp,offset,
                                                                dist_method, sub_pixel);*/
        cout << "matching end ||" << endl;

    }


}








}

