#pragma once

#define _USE_MATH_DEFINES
#include <math.h>

#include <Eigen/Core>
#include <iostream>

using namespace Eigen;
using namespace std;

namespace tpp
{
	inline float hanning(Vector3d xyz, Vector3d sizes)
	{
		const float hx = (sizes[0] == 1) ? 1.0 : 0.5 * (1.0 - cos(2 * M_PI * xyz[0] / (sizes[0] - 1)));
		const float hy = (sizes[1] == 1) ? 1.0 : 0.5 * (1.0 - cos(2 * M_PI * xyz[1] / (sizes[1] - 1)));
		const float hz = (sizes[2] == 1) ? 1.0 : 0.5 * (1.0 - cos(2 * M_PI * xyz[2] / (sizes[2] - 1)));
		return hx * hy * hz;
	}
}
