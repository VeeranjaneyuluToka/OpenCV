#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

using namespace std;
using namespace cv;

void find_ball_region(uchar *data, int width, int height){
	for (int r = 0; r < height; r++){
		for (int c = 0; c < width; c++){
			int ind = (r*width+c)*3;

			data[ind] = data[ind];
			data[ind+1] = data[ind+1];
			data[ind+2] = data[ind+2];
		}
	}
}

void hough_cir_exp(Mat frame){
	Mat gray;
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

	
	Mat frame, gray, roi, hsv_roi, mask;
	cap >> frame;
	/*
	cout << frame.size() << endl;
	uchar * data = frame.ptr();
	int width = frame.cols;
	int height = frame.rows;
	cout << width << " " << height << endl;
	find_ball_region(data, width, height);

	Mat nframe = Mat(height, width, CV_8UC3, data);
	imshow("frame", nframe);*/

        Rect track_window(162, 163, 35, 36);
        roi = frame(track_window);

        cvtColor(roi, hsv_roi, COLOR_BGR2HSV);
        inRange(hsv_roi, Scalar(0, 60, 32), Scalar(100, 255, 255), mask);

        float range_[] = {0, 180};
        const float* range[] = {range_};
        Mat roi_hist;
        int histSize[]={180};
        int channels[]={0};
        calcHist(&hsv_roi, 1, channels, mask, roi_hist, 1, histSize, range);
        normalize(roi_hist, roi_hist, 0, 255, NORM_MINMAX);

        TermCriteria term_crit(TermCriteria::EPS | TermCriteria::COUNT, 10, 1);

	while(true){
		Mat hsv, dst;
		char file_name[sizeof "./file_100.jpg"];

		cap>>frame;
		if (frame.empty())
			break;

		cvtColor(frame, hsv, COLOR_BGR2HSV);
		calcBackProject(&hsv, 1, channels, roi_hist, dst, range);

		meanShift(dst, track_window, term_crit);
		//sprintf(file_name, "./file_%03d.jpg", i);
		//imwrite(file_name, frame);

		rectangle(frame, track_window, 255, 2);

		imshow("frame", frame);	
		char c = (char)waitKey(25);
		if (c==27)
			break;
	}

	cap.release();
	destroyAllWindows();

	return 0;
}
