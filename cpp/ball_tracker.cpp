#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int find_ball_region(uchar * data, int width, int height){

}

int main(int argc, char** argv){
	string file_name;

	cout << argc << endl;
	for (int i = 0; i < argc; i++){
		if (i == 0)
			continue;

		file_name = argv[i];
	}
	cout << file_name << endl;

	VideoCapture cap(file_name);
	if (!cap.isOpened()){
		cout << "Error in opening video stream or file" << endl;
		return -1;
	}

	Mat frame, gray;
	cap >> frame;
	cout << frame.size() << endl;
	uchar * data = frame.ptr();
	int width = frame.cols;
	int height = frame.rows;
	cout << width << " " << height << endl;
	find_ball_region(data, width, height);
	/*
	cvtColor(frame, gray, COLOR_BGR2GRAY);
	GaussianBlur(gray, gray, Size(9, 9), 2, 2);
	vector<Vec3f> circles;
	HoughCircles(gray, circles, HOUGH_GRADIENT, 2, 50, 100, 60, 0, 1000);
	cout << circles.size()<<endl;
	for (int i = 0; i < circles.size(); i++){
		cout << circles[i][0] << " " << circles[i][1] << " " << circles[i][2] <<endl;
		int x = circles[i][0];
		int y = circles[i][1];
		int r = static_cast<int>(circles[i][2]);
		//circle(frame, Point(x, y), r, Scalar(0, 255, 0), 5);
	}
	imshow("gray", gray);
	imshow("frame", frame);
	waitKey(0);
	*/
	/*
	while(true){
		Mat frame;
		cap>>frame;
		if (frame.empty())
			break;

		imshow("frame", frame);
		char c = (char)waitKey(25);
		if (c==27)
			break;
	}*/

	cap.release();
	destroyAllWindows();

	return 0;
}
