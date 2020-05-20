
//#include <opencv2/opencv_modules.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <string>

using namespace std;
using namespace cv;

void display_frames(Mat img1, Mat img2){
	//namedWindow("Display_Window", WINDOW_AUTOSIZE);

	imshow("frame-1", img1);
	imshow("frame-2", img2);

	waitKey(0);
}

void compute_optical_flow(Mat img1, Mat img2, int iterations = 100, int avg_wnd=5, double alpha = 1){
	Mat gray_img1, gray_img2;

	cvtColor(img1, gray_img1, COLOR_BGR2GRAY);
	cvtColor(img2, gray_img2, COLOR_BGR2GRAY);
	//display_frames(gray_img1, gray_img2);
	
	Mat gray_img1_db, gray_img2_db;

	gray_img1.convertTo(gray_img1_db, CV_64FC1, 1.0/255.0);
	gray_img2.convertTo(gray_img2_db, CV_64FC1, 1.0/255.0);
	//display_frames(gray_img1_db, gray_img2_db);
	
	Mat I_t = gray_img2_db - gray_img1_db; //directional gradient
	//display_frames(I_t, gray_img2_db);
	
	Mat I_x, I_y;
	int ddepth = -1;

	Sobel(gray_img1_db, I_x, ddepth, 1, 0, 3); //x-direction gradient
	Sobel(gray_img2_db, I_y, ddepth, 0, 1, 3); //y-direction gradient
	//display_frames(I_x, I_y);
	
	Mat u = Mat::zeros(I_t.rows, I_t.cols, CV_64FC1);
	Mat v = Mat::zeros(I_t.rows, I_t.cols, CV_64FC1);
	//display_frames(u, v);
	
	Mat kernel = Mat::ones(avg_wnd, avg_wnd, CV_64FC1) / pow(avg_wnd, 2);
	//display_frames(kernel, v);
	
	for (int i = 0; i < iterations; i++){
		Mat u_avg, v_avg;

		Point anchor(kernel.cols - kernel.cols/2 -1, kernel.rows - kernel.rows/2 - 1);

		filter2D(u, u_avg, u.depth(), kernel, anchor, 0, BORDER_CONSTANT);
	       	filter2D(v, v_avg, v.depth(), kernel, anchor, 0, BORDER_CONSTANT);
		//display_frames(u, v);

		Mat C_prod1, C_prod2, I_x_squared, I_y_squared, I_x_C, I_y_C, C;

		multiply(I_x, u_avg, C_prod1);
		multiply(I_y, v_avg, C_prod2);
		multiply(I_x, I_x, I_x_squared);
		multiply(I_y, I_y, I_y_squared);

		Mat C_num = C_prod1 + C_prod2 + I_t;
		Mat C_den = pow(alpha, 2) + I_x_squared + I_y_squared;

		divide(C_num, C_den, C);

		multiply(I_x, C, I_x_C);
		multiply(I_y, C, I_y_C);

		u = u_avg - I_x_C;
		v = v_avg - I_y_C;
	}

	display_frames(u, v);
}

int main(int argc, char **argv){

	string img1_path = "/home/dell/Downloads/vuu/data/5454/99.jpg";
	string img2_path = "/home/dell/Downloads/vuu/data/5454/100.jpg";

	Mat img1 = imread(img1_path, IMREAD_COLOR);
	Mat img2 = imread(img2_path, IMREAD_COLOR);
	if (img1.empty() || img2.empty()){
		cout <<"image reading failed" << endl;
		return -1;
	}

	compute_optical_flow(img1, img2);

	//display_frames(img1, img2);

	return 0;
}
