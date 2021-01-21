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
	//from http://fastcpp.blogspot.fr/2011/06/bilinear-pixel-interpolation-using-sse.html
	template <typename T>
	struct Pixel
	{
		Pixel() :r(0), g(0), b(0), a(0), gray(0) {}
		Pixel(T gray) : gray(gray), r(0), g(0), b(0), a(0) {}
		Pixel(int rgba) { *((int*)this) = rgba; }
		Pixel(T r, T g, T b, T a) : r(r), g(g), b(b), a(a) {}
		Pixel(T r, T g, T b, T a, T gray) : r(r), g(g), b(b), a(a), gray(gray) {}
		T r, g, b, a, gray;
	};

	template<typename T>
	inline Pixel<T> bilinear(cv::Point2f current_pos, const cv::Mat img)
	{
		double x = current_pos.x;
		double y = current_pos.y;

		int px = (int)current_pos.x;
		int py = (int)current_pos.y;

		const int stride = img.cols;
		const Pixel<T> pix = Pixel<T>(img.at<uchar>(current_pos));

		// load the four neighboring pixels
		const Pixel<T> p1 = Pixel<T>(img.at<uchar>(current_pos - cv::Point2f(0, 1)));
		const Pixel<T> p2 = Pixel<T>(img.at<uchar>(current_pos - cv::Point2f(1, 0)));
		const Pixel<T> p3 = Pixel<T>(img.at<uchar>(current_pos + cv::Point2f(1, 0)));
		const Pixel<T> p4 = Pixel<T>(img.at<uchar>(current_pos + cv::Point2f(0, 1)));

		// Calculate the weights for each pixel
		float fx = x - px;
		float fy = y - py;
		float fx1 = 1.0f - fx;
		float fy1 = 1.0f - fy;

		T w1 = fx1 * fy1 * 256.0f;
		T w2 = fx  * fy1 * 256.0f;
		T w3 = fx1 * fy  * 256.0f;
		T w4 = fx  * fy  * 256.0f;

		// Calculate the weighted sum of pixels (for each color channel)
		T outr = p1.r * w1 + p2.r * w2 + p3.r * w3 + p4.r * w4;
		T outg = p1.g * w1 + p2.g * w2 + p3.g * w3 + p4.g * w4;
		T outb = p1.b * w1 + p2.b * w2 + p3.b * w3 + p4.b * w4;
		T outa = p1.a * w1 + p2.a * w2 + p3.a * w3 + p4.a * w4;
		T outgray = p1.gray * w1 + p2.gray * w2 + p3.gray * w3 + p4.gray * w4;

		//return Pixel<T>(outr >> 8, outg >> 8, outb >> 8, outa >> 8, outgray >> 8);

		return Pixel<T>(outr, outg, outb, outa, outgray);
	}

	void bilinear(cv::Point2f current_pos)
	{
		//assuming current_pos is where you are in the image

		//bilinear interpolation
		float dx = current_pos.x - (int)current_pos.x;
		float dy = current_pos.y - (int)current_pos.y;

		float weight_tl = (1.0 - dx) * (1.0 - dy);
		float weight_tr = (dx)       * (1.0 - dy);
		float weight_bl = (1.0 - dx) * (dy);
		float weight_br = (dx)       * (dy);
	}
}