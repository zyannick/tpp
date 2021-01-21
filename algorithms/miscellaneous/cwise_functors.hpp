#pragma once

#include <Eigen/Core>
#include <iostream>

#define _USE_MATH_DEFINES
#include <math.h>

using namespace Eigen;
using namespace std;

namespace tpp
{
	template<typename Scalar> struct mypow {
		EIGEN_STRONG_INLINE const Scalar operator() (const Scalar& _x, const Scalar& _y) const
		{
			using std::pow;
			Scalar res = pow(_x, _y);
			return res;
		}
	};

	template<typename Scalar> struct mymin {
		EIGEN_STRONG_INLINE const Scalar operator() (const Scalar& _x, const Scalar& _y) const
		{
			Scalar res;
			if (_x < _y)
				res = _x;
			else
				res = _y;
			return res;
		}
	};

	template<typename Scalar> struct mymax {
		EIGEN_STRONG_INLINE const Scalar operator() (const Scalar& _x, const Scalar& _y) const
		{
			Scalar res;
			if (_x > _y)
				res = _x;
			else
				res = _y;
			return res;
		}
	};

	template<typename Scalar> struct mymulpli {
		EIGEN_STRONG_INLINE const Scalar operator() (const Scalar& _x, const Scalar& _y) const
		{
			Scalar res = _x * _y;
			return res;
		}
	};

	template<typename Scalar> struct mydivide {
		EIGEN_STRONG_INLINE const Scalar operator() (const Scalar& _x, const Scalar& _y) const
		{
			Scalar res = _x / _y;
			return res;
		}
	};

	template<typename Scalar> struct mydivid_complex_float {
		EIGEN_STRONG_INLINE const Scalar operator() (const Scalar& _x, const float& _y) const
		{
			Scalar res = _x / _y;
			return res;
		}
	};

	template<typename Scalar> struct mymulpli_complex_float {
		EIGEN_STRONG_INLINE const Scalar operator() (const Scalar& _x, const float& _y) const
		{
			Scalar res = _x * _y;
			return res;
		}
	};

	template<typename Scalar> struct mymulpli_complex_complex {
		EIGEN_STRONG_INLINE const Scalar operator() (const Scalar& _x, const  Scalar& _y) const
		{
			Scalar res = _x * _y;
			return res;
		}
	};

	/*template<typename Scalar> struct my_magnitude {
	EIGEN_STRONG_INLINE const Scalar operator() (const std::complex<Scalar>& _x) const
	{
	using std::sqrt;
	using std::pow;
	Scalar res = _x.real /*= sqrt(pow(_x.real,2) + pow(_x.imag,2));
	return res;
	}
	};
	*/
	template<typename Scalar> struct mycosine {
		EIGEN_STRONG_INLINE const Scalar operator() (const Scalar& _x) const
		{
			using std::cos;
			Scalar res = cos(_x);
			return res;
		}
	};

	template<typename Scalar> struct mysine {
		EIGEN_STRONG_INLINE const Scalar operator() (const Scalar& _x) const
		{
			using std::sin;
			Scalar res = sin(_x);
			return res;
		}
	};

	template<typename Scalar> struct my_atan2 {
		EIGEN_STRONG_INLINE const Scalar operator() (const Scalar& _y, const Scalar& _x) const
		{
			using std::atan2;
			Scalar res = atan2(_y, _x);
			return res;
		}
	};

	template<typename Scalar> struct my_atan {
		EIGEN_STRONG_INLINE const Scalar operator() (const Scalar& _y) const
		{
			using std::atan;
			Scalar res = atan(_y);
			return res;
		}
	};

	template<typename Scalar> struct my_acos {
		EIGEN_STRONG_INLINE const Scalar operator() (const Scalar& _y) const
		{
			using std::acos;
			Scalar res = acos(_y);
			return res;
		}
	};

	template<typename Scalar> struct mylog {
		EIGEN_STRONG_INLINE const Scalar operator() (const Scalar& _x) const
		{
			using std::log;
			Scalar res = log(_x);
			return res;
		}
	};

	template<typename Scalar> struct myexpo {
		EIGEN_STRONG_INLINE const Scalar operator() (const Scalar& _x) const
		{
			using std::exp;
			Scalar res = exp(_x);
			return res;
		}
	};

	template<typename Scalar> struct myfloor {
		EIGEN_STRONG_INLINE const Scalar operator() (const Scalar& _x) const
		{
			using std::floor;
			Scalar res = floor(_x);
			return res;
		}
	};

	template<typename Scalar> struct my_wrap_angle_0_pi {
		EIGEN_STRONG_INLINE const Scalar operator() (const Scalar& _x) const
		{
			Scalar res = _x;
			if (_x < 0)
				res = _x + M_PI;
			return res;
		}
	};

	template<typename Scalar> struct my_fix {
		EIGEN_STRONG_INLINE const Scalar operator() (const Scalar& _x) const
		{
			Scalar res = _x;
			if (_x < 0)
				res = ceil(_x);
			else if (_x > 0)
				res = floor(_x);
			return res;
		}
	};

	struct set_real_part {
		EIGEN_STRONG_INLINE const std::complex<float> operator() (const std::complex<float>& _x, const float& _y) const
		{
			std::complex<float> res;
			float im = 0;
			res.imag(im);
			res.real(_y);
			return res;
		}
	};

    template<typename Scalar>
	struct set_imag_part {
		EIGEN_STRONG_INLINE const std::complex<Scalar> operator() (const std::complex<float>& _x, const float& _y) const
		{
			std::complex<float> res;
			res.real(_y);
			res.imag(_y);
			return res;
		}
	};

    template<typename Scalar> struct my_bin {
        EIGEN_STRONG_INLINE const Scalar operator() (const Scalar& _x, const Scalar _y) const
        {
            Scalar res;
            if(_x < _y )
                res = 0;
            else
                res = 1;
            return res;
        }
    };

}
