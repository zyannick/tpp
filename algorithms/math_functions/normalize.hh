#pragma once

void
normalize_min_max(float *array, size_t size, float new_min, float new_max)
{
	float old_min = FLT_MAX;
	float old_max = -FLT_MAX;

	// Find minimum and maximum.
	for (size_t i = 0; i < size; ++i) {
		if (old_min > array[i])
			old_min = array[i];
		if (old_max < array[i])
			old_max = array[i];
	}

	float old_range = old_max - old_min;
	float new_range = new_max - new_min;

	// Put data in the range [new_min, new_max].
	for (size_t i = 0; i < size; ++i)
		array[i] = (array[i] - old_min) / old_range;
	for (size_t i = 0; i < size; ++i)
		array[i] = array[i] * new_range + new_min;
}