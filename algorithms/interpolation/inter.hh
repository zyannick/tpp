#pragma once

#include <Eigen/Core>

using namespace Eigen;

MatrixXf createImage(int w, int h)
{
	MatrixXf  img = MatrixXf::Zero(h,w);
	//img = cvCreateImage(cvSize(w, h), IPL_DEPTH_8U, 3);
	return img;
}

MatrixXf bilinear__(MatrixXf img, int newWidth, int newHeight)
{
	int w = newWidth;
	int h = newHeight;
	MatrixXf img2 = MatrixXf::Zero(h, w);

	//uchar * Data = img2->imageData;
	//uchar * data = img->imageData;


	int a, b, c, d, x, y, index;
	float tx = (float)(img.cols() - 1) / w;
	float ty = (float)(img.rows() - 1) / h;


	float x_diff, y_diff;
	int i, j;

	for (i = 0; i<h; i++)
		for (j = 0; j<w; j++)
		{
			x = (int)(tx * j);
			y = (int)(ty * i);

			x_diff = ((tx * j) - x);
			y_diff = ((ty * i) - y);

			index = x + y * img.rows();
			// cac diem lan can
			a = (int)index;
			b = (int)(index + 1);
			c = (int)(index + img.cols());
			d = (int)(index + img.cols() + 1);
			img2[i*img2.cols() + j * 1 ] =
				img[a] * (1 - x_diff)*(1 - y_diff)
				+ img[b] * (1 - y_diff)*(x_diff)
				+ img[c] * (y_diff)*(1 - x_diff)
				+ img[d] * (y_diff)*(x_diff);
		}
	return img2;
}
