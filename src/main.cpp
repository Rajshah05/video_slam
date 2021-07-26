#include<iostream>
#include<opencv2/opencv.hpp>
#include"frame_output.h"
#include"featureExtractorAndMatcher.h"

const int W = 1920/2;
const int H = 1080/2;

const int MAX_FEATURES = 5000;
const float GOOD_MATCH_PERCENT = 0.15f;

FeatureExtractorAndMatcher fem;


void process_frame(cv::Mat& frame) {
	
	Frame_output Frame_o = fem.ExtractAndMatch(frame);

	// drawing keypoints on frame
	cv::Mat frame_kps;
	cv::drawKeypoints(frame, Frame_o.kps, frame_kps);

	cv::imshow("Frame", frame_kps);


}


int main(int argc, char** argv) {
	
	cv::VideoCapture cap("../video/test_countryroad.mp4");	
	if(!cap.isOpened()) {
		std::cout << "Error opening video stream or file" << '\n';
		return -1;
	}
	
	
	while(1) {
		cv::Mat frame, frame_rsz;
		Frame_output frame_o;
		cap >> frame;
		
		// Resizing frame
		cv::resize(frame, frame_rsz, cv::Size(W,H));
		
		if(frame.empty()) {
			break;
		}

		process_frame(frame_rsz);

		char c = (char)cv::waitKey(25);
		if(c == 27) break;
	}
	cap.release();
	cv::destroyAllWindows();
	return 0;
}

