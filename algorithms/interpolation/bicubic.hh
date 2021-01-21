#pragma once

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <list>
using namespace std;
// 

namespace tpp
{
	/* A possible bicubic interpolation consists in performing a convolution
	with the following kernel :
	[ 0  2  0  0][s0]
	[-1  0  1  0][s1]
	p(t) = 0.5 [1 t t^2 t^3][ 2 -5  4 -1][s2]
	[-1  3 -3  1][s3]
	Where s0..s3 are the samples, and t is the range from 0.0 to 1.0 */
	inline float bicubic(float s0, float s1, float s2, float s3, float t) {
		float r0 = 0.5f * (2.0f*s1);
		float r1 = 0.5f * (-s0 + s2);
		float r2 = 0.5f * (2.0*s0 - 5.0f*s1 + 4.0f*s2 - s3);
		float r3 = 0.5f * (-s0 + 3.0f*s1 - 3.0f*s2 + s3);
		return r3*t*t*t + r2*t*t + r1*t + r0;
	}

	/* sYX is a matrix with the samples (row-major)
	xt and yt are the interpolation fractions in the two directions ranging from
	0.0 to 1.0 */
	inline float bicubic2D(
		float s00, float s01, float s02, float s03,
		float s10, float s11, float s12, float s13,
		float s20, float s21, float s22, float s23,
		float s30, float s31, float s32, float s33, float xt, float yt) {
		// The bicubic convolution consists in passing the bicubic kernel in x
		// and then in y (or vice-versa, really)
		float r0 = bicubic(s00, s01, s02, s03, xt);
		float r1 = bicubic(s10, s11, s12, s13, xt);
		float r2 = bicubic(s20, s21, s22, s23, xt);
		float r3 = bicubic(s30, s31, s32, s33, xt);
		return bicubic(r0, r1, r2, r3, yt);
	}
}