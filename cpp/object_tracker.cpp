/*pexip_assignment.cpp : This file contains the 'main' function. Program execution begins and ends there.
This solution is implemented completely based on below reference
Reference[1]: "Kernel-based object tracking"
*/
#include "pexip_assignment.h"

using namespace std;
using namespace cv;

#define WND_SIZE 30
#define DELTA 15

ball_tracker::ball_tracker(uint w, uint h) {
    hist_bin_width = P_RNG / NUM_BINS;
    fw = w;
    fh = h;
}

ball_tracker::~ball_tracker() {
    delete[] target_buffer;
}

/*
Find ball reggion on first frame using sliding window approach
Note considered only red channel from BGR data as boll is in red color, this would elegent in terms of memory usage and performance
*/
void ball_tracker::find_target_position(uchar *bgr, uint width, uint height) {
    uchar* red_channel = new uchar[width*height];

    /*
    Extract red channel buffer from BGR buffer
    */
    for (uint i = 0; i < height; i++) {
        for (uint j = 0; j < width; j++) {
            red_channel[i * width + j] = bgr[((i * width + j) * 3) + 2]; //separate red channel from BGR buffer
        }
    }

    /*
    Have checked the ball region in a frame manually and assumed that as a probable kernel size to compute the target region.
    This can be further automated using the Region growing approach based Bredth first search.
    */
    uint max_val = 0;
    uint xInd = 0, yInd = 0;
    uint kw = WND_SIZE, kh = WND_SIZE;
    uint hkw = kw / 2, hkh = kh / 2;
    for (uint i = 0; i < (height-kh); i+=hkh) {
        for (uint j = 0; j < (width-kw); j+=hkw) {
            uint sum_wnd = 0;

            /*
            Accumulate red channel intensity values in the current window and check the global maxima 
            where large red regoin present which is what ball is
            */
            for (uint m = 0; m < kh; m++) {
                for (uint n = 0; n < kw; n++) {
                    sum_wnd += red_channel[(i + m) * width + (j + n)];
                }
            }
            if (sum_wnd > max_val) {
                max_val = sum_wnd;
                xInd = j;
                yInd = i;
            }
        }
    }

    //update ball region here
    target_pos.x = xInd;
    target_pos.y = yInd;
    target_pos.width = kw;
    target_pos.height = kh;

    delete[] red_channel; // cleanup memory
}

/*
Equation (12) from reference[1]
*/
float ball_tracker::Epanechnikov_kernel(float* buffer, int w, int h) {
    float epanechnikov_cd = 0.1 * PI * h * w;
    float kernel_sum = 0.0;

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            float x = static_cast<float>(i - h / 2);
            float  y = static_cast<float> (j - w / 2);

            float norm_x = x * x / (h * h / 4) + y * y / (w * w / 4);
            float result = norm_x < 1 ? (epanechnikov_cd * (1.0 - norm_x)) : 0;
            buffer[i * w + j] = result;
            kernel_sum += result;
        }
    }
    return kernel_sum;
}

/*
Compute the kernel
Build the histogram of G and R channel by quatizing the color information based on equation (13) in Reference[1]
*/
float* ball_tracker::target_pdf_representation(const uchar* bgr, int fw) {
    int w = target_pos.width;
    int h = target_pos.height;

    float* buffer = new float[w * h];
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            buffer[i * w + j] = 0.0;
        }
    }
    float norm_c = 1.0 / Epanechnikov_kernel(buffer, w, h);

    /*
    create a buffer to store the weighted quentized values of target
    This buffer should be deleted by callee of this function
    */
    float* pdf_model = new float[2 * 16];
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 16; j++) {
            pdf_model[i * 16 + j] = 1e-10;
        }
    }
    int g, r;
    g = r = 0;
    int ri = target_pos.y;
    int ci = target_pos.x;
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            g = bgr[((ri + i) * fw + (ci + j)) * 3 + 1] / hist_bin_width;
            r = bgr[((ri + i) * fw + (ci + j)) * 3 + 2] / hist_bin_width;

            pdf_model[g] += buffer[i * w + j] * norm_c;
            pdf_model[1*16+r] += buffer[i * w + j] * norm_c;
        }
    }

    delete[] buffer;

    return pdf_model;
}

/*
Initialize the target region to track
*/
void ball_tracker::initialize_target_region(const uchar* bgr, int fw) {
    target_buffer = target_pdf_representation(bgr, fw);
}

/*
Calculate the weight matrix based on the target location and current location PDFs
*/
float* ball_tracker::calcWeight(const uchar* bgr, float* target, float* current, int fw){
    int rows = target_pos.height;
    int cols = target_pos.width;

    float* weight = new float[rows * cols];
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            weight[i * cols + j] = 1.0000;

    int ri = target_pos.y;
    int ci = target_pos.x;

    int c=0;
    int channels = 2; //considering only Green and Red channels as blue does not influence much in this video
    for (int k = 0; k < channels; k++) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                c = bgr[((ri + i) * fw + (ci + j)) * 3 + (k+1)] / hist_bin_width;

                weight[i * cols + j] *= static_cast<float>((sqrt(target[k*NUM_BINS+c]) / current[k* NUM_BINS +c]));
            }
        }
    }

    return weight;
}

/*
Check the boundary cases to avoid the crashes in edge cases of target region computed
*/
void ball_tracker::check_region_bounds() {
    if (target_pos.x < 0)
        target_pos.x = 0;
    if (target_pos.y < 0)
        target_pos.y = 0;

    if (target_pos.width > this->fw)
        target_pos.width = this->fw;
    if (target_pos.height > this->fh)
        target_pos.height = this->fh;
}

/*
This the function where it finds the position of target in a new frame
*/
box ball_tracker::track_target_region(const uchar* bgr, int fw) {
    box nb;
    for (int i = 0; i < MAX_ITER; i++) {
        check_region_bounds();

        float* bgr_cand_hist = target_pdf_representation(bgr, fw);
        float* weight = calcWeight(bgr, target_buffer, bgr_cand_hist, fw);

        float delta_x = 0.0;
        float sum_wij = 0.0;
        float delta_y = 0.0;
        int rows = target_pos.height;
        int cols = target_pos.width;
        float centre = static_cast<float>((rows - 1) / 2.0);
        double mult = 0.0;

        // Find the next location of the target candidate according to (13): a simple weighted average from Reference[1]
        nb.x = target_pos.x;
        nb.y = target_pos.y;
        nb.width = target_pos.width;
        nb.height = target_pos.height;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                float norm_i = static_cast<float>(i - centre) / centre;
                float norm_j = static_cast<float>(j - centre) / centre;

                mult = pow(norm_i, 2) + pow(norm_j, 2) > 1.0 ? 0.0 : 1.0;
                delta_x += static_cast<float>(norm_j * weight[i*cols+j] * mult);
                delta_y += static_cast<float>(norm_i * weight[i*cols+j] * mult);
                sum_wij += static_cast<float>(weight[i*cols+j] * mult);
            }
        }

        nb.x += static_cast<int>((delta_x / sum_wij) * centre);
        nb.y += static_cast<int>((delta_y / sum_wij) * centre);

        if (abs(nb.x - target_pos.x) < 1 && abs(nb.y - target_pos.y) < 1) {
            break;
        }
        else {
            target_pos.x = nb.x;
            target_pos.y = nb.y;
        }

        delete[] bgr_cand_hist;
        delete[] weight;
    }
    return nb;
}

Point ball_tracker::get_point() {
    Point pt;
    pt.x = target_pos.x;
    pt.y = target_pos.y;
    return pt;
}

int main(){
    Mat frame;
    box ms_bbrect;
    Rect rect;
    Point ipt, npt;
    string vid_file_path = "D:/data/pexip/checkerball.mp4";

    VideoCapture cap(vid_file_path);
    if (!cap.isOpened()) {
        cout << "vido file opening failed" << endl;
        return -1;
    }

    /*
    Read first frame
    Compute the target position based on sliding window
    Initialize the target region by computing PDF
    */
    cap >> frame;
    ball_tracker bt(frame.cols, frame.rows);
    VideoWriter video("./ball_tracker_out.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, Size(frame.cols, frame.rows));
    bt.find_target_position(frame.data, frame.cols, frame.rows);
    ipt = bt.get_point();
    bt.initialize_target_region(frame.data, frame.cols);
    for (;;) {
        cap >> frame;
        if (frame.empty())
            break;

        /*
        Compute the next target probable region
        Compute the posision of it and update
        */
        ms_bbrect = bt.track_target_region(frame.data, frame.cols);
        rect.x = ms_bbrect.x;
        rect.y = ms_bbrect.y;
        rect.width = ms_bbrect.width;
        rect.height = ms_bbrect.height;
        npt = bt.get_point();
        ipt.x += DELTA;
        ipt.y += DELTA;
        npt.x += DELTA;
        npt.y += DELTA;
        line(frame, ipt, npt, Scalar(255, 0, 0), 3);
        ipt = npt;

        rectangle(frame, rect, Scalar(0, 255, 0), 3);

        video.write(frame);
        imshow("ms-demo", frame);
        char c = (char)waitKey(10);
        if (c == 27)
            break;
    }
    cap.release();
    video.release();
    destroyAllWindows();

    return 0;
}