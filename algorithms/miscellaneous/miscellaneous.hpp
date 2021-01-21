#pragma once

#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include "cwise_functors.hpp"
#include <random>
#ifdef vpp
    #include <vpp/vpp.hh>
    using namespace vpp;
#endif

#define _USE_MATH_DEFINES
#include <math.h>

 
using namespace Eigen;
using namespace std;




namespace tpp
{

std::vector<string> split(const string& str, const string& delim)
{
    std::vector<string> tokens;
    size_t prev = 0, pos = 0;
    do
    {
        pos = str.find(delim, prev);
        if (pos == string::npos) pos = str.length();
        string token = str.substr(prev, pos-prev);
        if (!token.empty()) tokens.push_back(token);
        prev = pos + delim.length();
    }
    while (pos < str.length() && prev < str.length());
    return tokens;
}


Vector3i hsv_to_rgb(float h, float s, float v) {
    float fC = v * s; // Chroma
    float fHPrime = fmod(h / 60.0, 6);
    float fX = fC * (1 - fabs(fmod(fHPrime, 2) - 1));
    float fM = v - fC;

    float r;
    float g;
    float b ;

    if (0 <= fHPrime && fHPrime < 1) {
        r = fC;
        g = fX;
        b = 0;
    }
    else if (1 <= fHPrime && fHPrime < 2) {
        r = fX;
        g = fC;
        b = 0;
    }
    else if (2 <= fHPrime && fHPrime < 3) {
        r = 0;
        g = fC;
        b = fX;
    }
    else if (3 <= fHPrime && fHPrime < 4) {
        r = 0;
        g = fX;
        b = fC;
    }
    else if (4 <= fHPrime && fHPrime < 5) {
        r = fX;
        g = 0;
        b = fC;
    }
    else if (5 <= fHPrime && fHPrime < 6) {
        r = fC;
        g = 0;
        b = fX;
    }
    else {
        r = 0;
        g = 0;
        b = 0;
    }

    r += fM;
    g += fM;
    b += fM;

    return Vector3i(b, g, r);
}

int sign_of_number(int number)
{
    if (number < 0)
        return -1;
    else
        return 1;
}


inline
Vector3i generate_color()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(5, 2);
    float h = d(gen)*M_PI;
    float s = d(gen)*M_PI;
    float v = d(gen)*M_PI;
    Vector3i colr;
    colr = hsv_to_rgb(h, s, v);
    int r, g, b;
    b = (int)colr[0];
    g = (int)colr[1];
    r = (int)colr[2];
    if (r + g + b < 200)
    {
        std::normal_distribution<> dn(10, 5);
        b = 10 * dn(gen);
        g = 10 * dn(gen);
        r = 10 * dn(gen);
        b = sign_of_number(b)*b;
        g = sign_of_number(g)*g;
        r = sign_of_number(r)*r;
        colr = Vector3i(b, g, r);
    }
    return colr;
}

inline
Vector3i generate_color(int th)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(5, 2);
    float h = d(gen)*M_PI;
    float s = d(gen)*M_PI;
    float v = d(gen)*M_PI;
    Vector3i colr;
    colr = hsv_to_rgb(h, s, v);
    int r, g, b;
    b = (int)colr[0];
    g = (int)colr[1];
    r = (int)colr[2];
    while (r + g + b < th)
    {
        std::normal_distribution<> dn(10, 5);
        b = 10 * dn(gen);
        g = 10 * dn(gen);
        r = 10 * dn(gen);
        b = sign_of_number(b)*b;
        g = sign_of_number(g)*g;
        r = sign_of_number(r)*r;
        colr = Vector3i(b, g, r);
    }
    return colr;
}


inline
float median_vector(VectorXf vec)
{
    float med;
    using std::sort;
    std::sort(vec.data(), vec.data() + vec.size());
    int n = vec.rows();
    if (n % 2 == 1)
    {
        med = vec[(n-1)/2];
    }
    else
    {
        med = (vec[floor(n / 2) - 1] + vec[floor(n / 2) ]) / 2;
    }
    return med;
}

inline
MatrixXf pow_matrix_coefficients(MatrixXf M, float power)
{
    MatrixXf N = MatrixXf::Zero(M.rows(), M.cols());
    for (int row = 0; row < M.rows(); row++)
    {
        for (int col = 0; col < M.cols(); col++)
        {
            N(row, col) = pow(M(row, col), power);
        }
    }
    return N;
}

inline
MatrixXf sqrt_matrix_coefficients(MatrixXf M)
{
    MatrixXf N = MatrixXf::Zero(M.rows(), M.cols());
    for (int row = 0; row < M.rows(); row++)
    {
        for (int col = 0; col < M.cols(); col++)
        {
            N(row, col) = sqrt(M(row, col));
        }
    }
    return N;
}

inline
MatrixXf atan2_matrix_coefficients(MatrixXf y, MatrixXf x)
{
    MatrixXf N = MatrixXf::Zero(y.rows(), y.cols());
    for (int row = 0; row < y.rows(); row++)
    {
        for (int col = 0; col < y.cols(); col++)
        {
            N(row, col) = atan2(y(row, col), x(row, col));
        }
    }
    return N;
}


template<typename Derived>
void printFirstRow(const Eigen::MatrixBase<Derived>& x)
{
  cout << x.row(0) << endl;
}


template<typename T>
inline
Matrix<T, Dynamic,1 > fftshift_vector(Matrix<T, Dynamic,1 > X)
{
    Matrix<T, Dynamic,1 > Y;
    int n;
    if (X.cols() == 1)
    {
        Y = Matrix<T, Dynamic,1 >::Zero(X.rows(),1);
        n = X.rows();
    }
    else if (X.rows() == 1)
    {
        Y = Matrix<T, Dynamic,1 >::Zero(X.cols(),1);
        n = X.cols();
    }

    if (n % 2 == 0)
    {
        int mid = n / 2;
        Y.segment(0, mid) = X.segment(mid, mid);
        Y.segment(mid, mid) = X.segment(0, mid);
    }
    else
    {
        int mid = int(floor(n / 2.0));
        //cout << "mid " << mid << endl;
        Y.segment(0, mid) = X.segment(mid + 1, mid );
        Y.segment(mid, mid + 1) = X.segment(0, mid +1 );
    }
    return Y;
}


#ifdef vpp


template<typename T>
inline
image2d<T> ifftshift_vector(image2d<T> X)
{
    Matrix<T, Dynamic,1 > Y;
    int n;
    if (X.cols() == 1)
    {
        Y = Matrix<T, Dynamic,1 >::Zero(X.rows(),1);
        n = X.rows();
    }
    else if (X.rows() == 1)
    {
        Y = Matrix<T, Dynamic,1 >::Zero(X.cols(),1);
        n = X.cols();
    }

    if (n % 2 == 0)
    {
        int mid = n / 2;
        Y.segment(0, mid) = X.segment(mid, mid);
        Y.segment(mid, mid) = X.segment(0, mid);
    }
    else
    {
        int mid = int(floor(n / 2.0));
        //cout << "mid " << mid << endl;
        Y.segment(0, mid + 1) = X.segment(mid, mid + 1);
        Y.segment(mid + 1, mid) = X.segment(0, mid);
    }
    return Y;
}
#endif

template<typename T>
inline
Matrix<T, Dynamic,1 > ifftshift_vector(Matrix<T, Dynamic,1 > X)
{
    Matrix<T, Dynamic,1 > Y;
    int n;
    if (X.cols() == 1)
    {
        Y = Matrix<T, Dynamic,1 >::Zero(X.rows(),1);
        n = X.rows();
    }
    else if (X.rows() == 1)
    {
        Y = Matrix<T, Dynamic,1 >::Zero(X.cols(),1);
        n = X.cols();
    }

    if (n % 2 == 0)
    {
        int mid = n / 2;
        Y.segment(0, mid) = X.segment(mid, mid);
        Y.segment(mid, mid) = X.segment(0, mid);
    }
    else
    {
        int mid = int(floor(n / 2.0));
        //cout << "mid " << mid << endl;
        Y.segment(0, mid + 1) = X.segment(mid, mid + 1);
        Y.segment(mid + 1, mid) = X.segment(0, mid);
    }
    return Y;
}

template<typename T>
inline
Matrix<T, Dynamic,Dynamic > fftshift_matrix(Matrix<T, Dynamic,Dynamic > X)
{
    Matrix<T, Dynamic,Dynamic > Y = Matrix<T, Dynamic,Dynamic >::Zero(X.rows(), X.cols());
    int n = X.rows();

    if (n % 2 == 0)
    {
        int mid = n / 2;
        for (int i = 0; i < mid; i++)
        {
            Y.row(i) = fftshift_vector<T>(X.row(mid + i));
            Y.row(mid + i) = fftshift_vector<T>(X.row(i));
        }
    }
    else
    {
        int mid = int(floor(n / 2.0));
        for (int i = 0; i < mid; i++)
        {
            Y.row(i) = fftshift_vector<T>(X.row(mid + i + 1));
        }
        for (int i = 0; i <= mid; i++)
        {
            Y.row(mid + i) = fftshift_vector<T>(X.row(i));
        }
    }
    return Y;
}


/*
inline
VectorXcf fftshift_vector(VectorXcf X)
{
    VectorXcf Y;
    int n;
    if (X.cols() == 1)
    {
        Y = VectorXcf::Zero(X.rows());
        n = X.rows();
    }
    else if (X.rows() == 1)
    {
        Y = VectorXcf::Zero(X.cols());
        n = X.cols();
    }

    if (n % 2 == 0)
    {
        int mid = n / 2;
        Y.segment(0, mid) = X.segment(mid, mid);
        Y.segment(mid, mid) = X.segment(0, mid);
    }
    else
    {
        int mid = int(floor(n / 2.0));
        //cout << "mid " << mid << endl;
        Y.segment(0, mid) = X.segment(mid + 1, mid );
        Y.segment(mid, mid + 1) = X.segment(0, mid +1 );
    }
    return Y;
}

inline
MatrixXcf fftshift_matrix(MatrixXcf X)
{
    MatrixXcf Y = MatrixXcf::Zero(X.rows(), X.cols());
    int n = X.rows();

    if (n % 2 == 0)
    {
        int mid = n / 2;
        for (int i = 0; i < mid; i++)
        {
            Y.row(i) = fftshift_vector(X.row(mid + i));
            Y.row(mid + i) = fftshift_vector(X.row(i));
        }
    }
    else
    {
        int mid = int(floor(n / 2.0));
        for (int i = 0; i < mid; i++)
        {
            Y.row(i) = fftshift_vector(X.row(mid + i + 1));
        }
        for (int i = 0; i <= mid; i++)
        {
            Y.row(mid + i) = fftshift_vector(X.row(i));
        }
    }
    return Y;
}/**/

template <typename T>
inline
Matrix<T, Dynamic,Dynamic > ifftshift_matrix(Matrix<T, Dynamic,Dynamic > X)
{
    Matrix<T, Dynamic,Dynamic > Y = Matrix<T, Dynamic,Dynamic >::Zero(X.rows(), X.cols());
    int n = X.rows();

    if (n % 2 == 0)
    {
        int mid = n / 2;
        for (int i = 0; i < mid; i++)
        {
            //cout << " row " << X.row(mid + i) << endl;
            Y.row(i) = ifftshift_vector<T>(X.row(mid + i));
            Y.row(mid + i) = ifftshift_vector<T>(X.row(i));
        }
    }
    else
    {
        int mid = int(floor(n / 2.0));
        //cout << "ceil " << n / 2.0 << endl;
        //cout << "mid " << mid << endl;
        for (int i = 0; i <= mid; i++)
        {
            //cout << "val " << i << " p " << mid + i << endl;
            Y.row(i) = ifftshift_vector<T>(X.row(mid + i));
        }
        for (int i = 0; i < mid; i++)
        {
            //cout << "vol " << mid + i << endl;
            Y.row(mid + 1 + i) = ifftshift_vector<T>(X.row(i));
        }
    }
    return Y;
}

inline
MatrixXf cosine_matrix(MatrixXf M)
{
    MatrixXf Result = MatrixXf::Zero(M.rows(), M.cols());
    for (int row = 0; row < M.rows(); row++)
    {
        for (int col = 0; col < M.cols(); col++)
        {
            Result(row, col) = cos(M(row, col));
        }
    }
}

inline
MatrixXf sine_matrix(MatrixXf M)
{
    MatrixXf Result = MatrixXf::Zero(M.rows(), M.cols());
    for (int row = 0; row < M.rows(); row++)
    {
        for (int col = 0; col < M.cols(); col++)
        {
            Result(row, col) = sin(M(row, col));
        }
    }
}

inline
VectorXf range_number_pair(int n)
{
    VectorXf result(n);
    for (float i = -n / 2.0,  k = 0; i <= (n / 2.0 - 1); i++, k++)
    {
        result[k] = float(i);
    }
    return result;
}

inline
VectorXf range_number_impair(int n)
{
    VectorXf result(n);
    for (float i = -(n - 1) / 2.0, k = 0; i <= (n - 1) / 2.0; i++, k++)
    {
        result[k] = (float)i;
    }
    return result;
}

inline
void mesgrid(VectorXf x, VectorXf y, MatrixXf &X, MatrixXf &Y)
{
    X.resize(y.rows(), x.rows());
    Y.resize(y.rows(), x.rows());
    //cout << "y " << y.rows() << "  x  " << x.rows() << endl;
    for (int row = 0; row < y.rows(); row++)
    {
        for (int col = 0; col < x.rows(); col++)
        {
            X(row, col) = x(col);
            Y(row, col) = y(row);
        }
    }
}

inline
void mesgrid_para(VectorXf x, VectorXf y, MatrixXf &X, MatrixXf &Y)
{
    X.resize(y.rows(), x.rows());
    Y.resize(y.rows(), x.rows());
    //cout << "y " << y.rows() << "  x  " << x.rows() << endl;
    for (int row = 0; row < y.rows(); row++)
    {
        for (int col = 0; col < x.rows(); col++)
        {
            X(row, col) = x(col);
            Y(row, col) = y(row);
        }
    }
}


#ifdef vpp
template <typename Type>
inline
void mesgrid_vpp(VectorXf x, VectorXf y, image2d<Type> &X, image2d<Type> &Y)
{
    X = image2d<Type>(y.rows(), x.rows());
    Y = image2d<Type>(y.rows(), x.rows());

    pixel_wise(X, Y, Y.domain()) | [&] (auto& i, auto &j, auto coord)
    {
        auto row = coord[0];
        auto col = coord[1];
        i = x(col);
        j = y(row);
    };

}
#endif

inline
void filter_grid(int nrows, int ncols, MatrixXf &radius, MatrixXf &u1, MatrixXf &u2)
{
    // Set up X and Y spatial frequency matrices, u1 and u2 The following code
    // adjusts things appropriately for odd and even values of rows and columns
    // so that the 0 frequency point is placed appropriately.See
    // https://blogs.uoregon.edu/seis/wiki/unpacking-the-matlab-fft/
    VectorXf ulrange;
    VectorXf u2range;
    MatrixXf matrix_two = MatrixXf::Constant(nrows, ncols, 2);
    if (ncols % 2)
        ulrange = range_number_impair(ncols) / ncols;
    else
        ulrange = range_number_pair(ncols) / ncols;

    if (nrows % 2)
        u2range = range_number_impair(nrows) / nrows;
    else
        u2range = range_number_pair(nrows) / nrows;

    mesgrid(ulrange, u2range, u1, u2);

    u1 = ifftshift_matrix(u1);
    u2 = ifftshift_matrix(u2);

    radius = (u1.cwiseAbs2() + u2.cwiseAbs2()).cwiseSqrt();
}

inline
MatrixXf low_pass_filter(int nrows, int ncols, float cut_off, int n)
{
    MatrixXf Result;
    assert(cut_off > 0 && cut_off < 0.5 && "cutoff frequency must be between 0 and 0.5");
    MatrixXf radius, u1, u2, mat_temp;
    filter_grid(nrows, ncols, radius, u1, u2);
    MatrixXf matrix_two_n = MatrixXf::Constant(nrows, ncols, 2 * n);
    mat_temp = radius / cut_off;
    mat_temp = mat_temp.binaryExpr(matrix_two_n, mypow<float>());
    mat_temp = mat_temp + MatrixXf::Ones(nrows, ncols);
    Result = mat_temp.cwiseInverse();
    return Result;
}

inline
int count_hist(VectorXf vec, double min, double max)
{
    int ct = 0;
    for (int row = 0; row < vec.rows(); row++)
    {
        if (vec[row] >= min && vec[row] <= max)
        {
            ct++;
        }
    }
    return ct;
}

inline
float rayleigh_mode(VectorXf M, int nbins = 50)
{
    double max = M.maxCoeff();
    int taille = nbins + 1;
    double i = 0;
    double over = (double)max + (double)max / (double)nbins;
    cout << "taille " << taille << " over " << over << " max " << max << endl;
    VectorXf edges = VectorXf::Zero(taille);
    edges(0) = 0;
    i = 1;
    while (i < taille)
    {
        edges[i] = edges[i - 1] + (double)max / (double)nbins;
        i++;
    }
    VectorXf hist = VectorXf::Zero(taille - 1);
    for (int row = 0, idx = 0; row < taille - 1; row++, idx++)
    {
        hist[idx] = count_hist(M, edges[row], edges[row + 1]);
    }
    MatrixXf::Index maxRow, maxCol;
    float max_value = hist.maxCoeff(&maxRow, &maxCol);
    float rmode = (edges(maxRow) + edges(maxRow + 1)) / 2;
    cout << "pkay " << endl;
    return rmode;
}


inline
VectorXf eigen_matrix_to_vector(MatrixXf mat)
{
    int nrows = mat.rows(), ncols = mat.cols();
    VectorXf vect = VectorXf::Zero(nrows*ncols);
    for(int row = 0, i = 0; row < nrows; row ++ )
        for(int col = 0 ; col < ncols ; col++, i++)
            vect(i) = mat(row,col);
    return vect;
}

/*
    */

inline
MatrixXf mat_to_eigen(cv::Mat img)
{
    //cout << "mat_to_eigen start" << endl;
    MatrixXf mat = MatrixXf::Zero(img.rows, img.cols);
    for (int row = 0; row < img.rows; row++)
    {
        for (int col = 0; col < img.cols; col++)
        {
            mat(row, col) = img.at<uchar>(row, col);
        }
    }
    //cout << "mat_to_eigen end" << endl;
    return mat;
}


inline
MatrixXf dilate(MatrixXf img)
{
    //cout << "mat_to_eigen start" << endl;
    MatrixXf mat = MatrixXf::Zero(img.rows(), img.cols());
    for (int row = 0; row < img.rows(); row++)
    {
        for (int col = 0; col < img.cols(); col++)
        {
            //mat(row, col) = img.at<uchar>(row, col);
        }
    }
    //cout << "mat_to_eigen end" << endl;
    return mat;
}

inline
int count_number_of_no_zero_pixels(cv::Mat img)
{
    int number = 0;
    for (int row = 0; row < img.rows; row++)
    {
        for (int col = 0; col < img.cols; col++)
        {
            if(img.at<uchar>(row, col) > 0)
            {
                number++;
            }
        }
    }
    return number;
}


cv::Mat eigen_to_mat(MatrixXf img, int channel = 1)
{
    //cout << "eigen_to_mat " << endl;
    cv::Mat mat;
    if(channel == 1)
    {
        //cout << "creating " << endl;
        mat = cv::Mat(img.rows(), img.cols(), CV_8U);
        //cout << "created " << endl;
    }
    else if(channel == 3)
        mat = cv::Mat(img.rows(), img.cols(), CV_8UC3);
    //cout << "hrer "  << endl;
    for (int row = 0; row < img.rows(); row++)
    {
        for (int col = 0; col < img.cols(); col++)
        {
            if(channel == 1)
            {
                mat.at<uchar>(row, col) = uchar(int(img(row, col)));
                //cout << int(img(row, col)) << endl;
            }
            else
            {
                uchar val = uchar(int(img(row, col)));
                mat.at<cv::Vec3b>(row, col) = cv::Vec3b(val, val, val );
            }
        }
    }
    return mat;
}

template< typename T>
inline
cv::Mat eigen_to_mat_template_1d(Matrix<T, Dynamic, Dynamic> img, int channel = 1)
{
    cv::Mat mat;
    if(channel == 1)
        mat = cv::Mat(img.rows(), img.cols(), CV_8U);
    else if(channel == 3)
        mat = cv::Mat(img.rows(), img.cols(), CV_8UC3);
    for (int row = 0; row < img.rows(); row++)
    {
        for (int col = 0; col < img.cols(); col++)
        {
            if(channel == 1)
                mat.at<uchar>(row, col) = uchar(int(img(row, col)));
            else
            {
                uchar val = uchar(int(img(row, col)));
                mat.at<cv::Vec3b>(row, col) = cv::Vec3b(val, val, val );
            }
        }
    }
    return mat;
}

template <typename T>
inline
cv::Mat eigen_to_mat_template(Matrix<T, Dynamic,Dynamic > img)
{
    int channels = img(0,0).rows();
    cv::Mat mat;
    if(channels == 1)
        mat = cv::Mat(img.rows(), img.cols(), CV_8U);
    else if(channels == 3)
        mat = cv::Mat(img.rows(), img.cols(), CV_8UC3);
    for (int row = 0; row < img.rows(); row++)
    {
        for (int col = 0; col < img.cols(); col++)
        {
            if(channels == 1)
                mat.at<uchar>(row, col) = uchar(img(row, col)(0));
            else
            {
                uchar r = uchar( (img(row, col))(0) );
                uchar g = uchar( (img(row, col))(1) );
                uchar b = uchar( (img(row, col))(2) );
                mat.at<cv::Vec3b>(row, col) = cv::Vec3b(r, g, b );
            }
        }
    }
    return mat;
}




inline
cv::Mat eigen_to_mat_float(MatrixXf img, int channel = 1)
{
    cv::Mat mat = cv::Mat(img.rows(), img.cols(), CV_64F);
    for (int row = 0; row < img.rows(); row++)
    {
        for (int col = 0; col < img.cols(); col++)
        {
            mat.at<float>(row, col) = img(row, col);
        }
    }
    //cout << "poopo " << mat << endl;

    return mat;
}


void save_vector(VectorXf vect_)
{
    ofstream my_file;
    my_file.open("vector.txt");

    my_file << " vector = [ ";

    for (int row = 0; row < vect_.rows(); row++)
    {
        my_file << vect_[row] << "  ";
    }
    my_file << " ]; \n";
}

void binarize_image(MatrixXf src, MatrixXf &dst , float th , float min_value = 0, float max_value = 1)
{
    int nrows = src.rows(), ncols = src.cols();
    dst = MatrixXf::Zero(nrows, ncols);
    int nb = 0;
    for (int row = 0; row < nrows; row++)
    {
        for (int col = 0; col < ncols; col++)
        {
            if( src(row,col) > th)
            {
                dst(row,col) = max_value;
                nb++;
            }
            else
            {
                dst(row,col) = min_value;
            }
        }
    }
}


/**
 * @brief normalize_matrix
 * @param src
 * Normalize the values between 0 and 1
 */
void normalize_matrix(MatrixXf &src)
{
    //get location of maximum
    MatrixXf::Index maxRow_r, maxCol_r;
    float max_value = src.maxCoeff(&maxRow_r, &maxCol_r);
    //get location of minimum
    MatrixXf::Index minRow_r, minCol_r;
    float min_value = src.minCoeff(&minRow_r, &minCol_r);

    int nrows = src.rows(), ncols = src.cols();
    for (int row = 0; row < nrows; row++)
    {
        for (int col = 0; col < ncols; col++)
        {
            src(row,col) = ( src(row,col) - min_value ) / (max_value - min_value);
        }
    }
}

/**
 * @brief normalize_matrix
 * @param src
 * Normalize the values between 0 and 1
 */
template <typename T>
void normalize_matrix_template(Matrix<T, Dynamic , Dynamic> &src)
{
    //get location of maximum
    MatrixXf::Index maxRow_r, maxCol_r;
    T max_value = src.maxCoeff(&maxRow_r, &maxCol_r);
    //get location of minimum
    MatrixXf::Index minRow_r, minCol_r;
    T min_value = src.minCoeff(&minRow_r, &minCol_r);

    int nrows = src.rows(), ncols = src.cols();
    for (int row = 0; row < nrows; row++)
    {
        for (int col = 0; col < ncols; col++)
        {
            src(row,col) = T( ( src(row,col) - min_value ) / (max_value - min_value) );
        }
    }
}



void normalize_matrix_and_binarize(MatrixXf &src,float min_,float max_)
{
    //get location of maximum
    MatrixXf::Index maxRow_r, maxCol_r;
    float max_value = src.maxCoeff(&maxRow_r, &maxCol_r);
    //get location of minimum
    MatrixXf::Index minRow_r, minCol_r;
    float min_value = src.minCoeff(&minRow_r, &minCol_r);

    int nrows = src.rows(), ncols = src.cols();
    for (int row = 0; row < nrows; row++)
    {
        for (int col = 0; col < ncols; col++)
        {
            float val = ( src(row,col) - min_value ) / (max_value - min_value);
            src(row,col) = val > min_ ? max_ : min_;
        }
    }

}


/**
 * @brief normalize_matrix
 * @param src
 * @param max_intensity
 * Normalize the values between 0 and max_intensity
 */
void normalize_matrix(MatrixXf &src,  float max_intensity)
{
    int nrows = src.rows(), ncols = src.cols();

    for (int row = 0; row < nrows; row++)
    {
        for (int col = 0; col < ncols; col++)
        {
            if(src(row,col) < 0.3)
                src(row,col) = 0;
        }
    }

    //get location of maximum
    MatrixXf::Index maxRow_r, maxCol_r;
    float max_value = src.maxCoeff(&maxRow_r, &maxCol_r);
    //get location of minimum
    MatrixXf::Index minRow_r, minCol_r;
    float min_value = src.minCoeff(&minRow_r, &minCol_r);

    for (int row = 0; row < nrows; row++)
    {
        for (int col = 0; col < ncols; col++)
        {
            src(row,col) = max_intensity*( src(row,col) - min_value ) / (max_value - min_value);
        }
    }
}


void normalize_matrix(MatrixXf &src,  float max_intensity, float min_intensity)
{
    int nrows = src.rows(), ncols = src.cols();


    float max_value = max_intensity;

    float min_value = min_intensity;

    for (int row = 0; row < nrows; row++)
    {
        for (int col = 0; col < ncols; col++)
        {
            src(row,col) = ( src(row,col) - min_value ) / (max_value - min_value);
        }
    }
}

/**
 * @brief normalize_matrix_minus_max
 * @param src
 * @param max_intensity
 * Normalize the values between 0 and max_intensity (inverse)
 */
void normalize_matrix_minus_max(MatrixXf &src,  float max_intensity)
{
    int nrows = src.rows(), ncols = src.cols();

    //get location of maximum
    MatrixXf::Index maxRow_r, maxCol_r;
    float max_value = src.maxCoeff(&maxRow_r, &maxCol_r);
    //get location of minimum
    MatrixXf::Index minRow_r, minCol_r;
    float min_value = src.minCoeff(&minRow_r, &minCol_r);

    for (int row = 0; row < nrows; row++)
    {
        for (int col = 0; col < ncols; col++)
        {
            src(row,col) = max_intensity - max_intensity*( src(row,col) - min_value ) / (max_value - min_value);
        }
    }
}


template <typename T>
inline
Matrix<T, Dynamic,Dynamic > non_max_suppression(Matrix<T, Dynamic,Dynamic > source)
{
    Matrix<T, Dynamic,Dynamic > Result = Matrix<T, Dynamic,Dynamic >::Zero(source.rows(), source.cols());

    return Result;

}


struct ascending_depth
{
    // 3d points are in the format z, y, x
    inline bool operator() (const Vector3d& point3d1, const Vector3d& point3d2)
    {
        return (point3d1[0] < point3d2[0]);
    }
};




}
