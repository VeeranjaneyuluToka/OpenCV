#pragma once
#ifndef PEXIPASSIGNMENT_H
#define PEXIPASSIGNMENT_H

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

#define MAX_ITER 10
#define NUM_BINS 16
#define P_RNG 256
#define PI 3.1415926

struct box {
	int x;
	int y;
	int width;
	int height;
};

class ball_tracker {
public:
	ball_tracker(uint fw, uint fh);
	~ball_tracker();

	void find_target_position(uchar* bgr, uint width, uint height);
	void initialize_target_region(const uchar* bgr, int fw);
	box track_target_region(const uchar* bgr, int fw);
	cv::Point get_point();

private:
	float Epanechnikov_kernel(float* buffer, int w, int h);
	float* target_pdf_representation(const uchar* bgr, int fw);
	float* calcWeight(const uchar* bgr, float* target, float* current, int fw);
	void check_region_bounds();

	float hist_bin_width;
	box target_pos;
	float* target_buffer;
	uint fw, fh;
};

#endif // PEXIPASSIGNMENT_H

