#include<iostream>
#include<SDL2/SDL.h>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>

int main(int argc, char** argv) {
	
	cv::VideoCapture cap("../video/test_countryroad.mp4");	
	if(!cap.isOpened()) {
		std::cout << "Error opening video stream or file" << '\n';
		return -1;
	}
	while(1) {
		cv::Mat frame;
		cap >> frame;

		if(frame.empty()) {
			break;
		}
		cv::imshow("Frame", frame);

		char c = (char)cv::waitKey(25);
		if(c == 27) break;
	}
	cap.release();
	cv::destroyAllWindows();
	return 0;
}

