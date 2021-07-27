#include<iostream>
#include<opencv2/opencv.hpp>
#include"featureExtractorAndMatcher.h"

const int W = 1920/2;
const int H = 1080/2;

FeatureExtractorAndMatcher fem;


void process_frame(cv::Mat& frame) {
	
	std::vector<match_kp> mkps = fem.ExtractAndMatch(frame);

	if(!mkps.size()) return;
	// drawing keypoints on frame
	// cv::Mat frame_mkps;
	std::vector<cv::KeyPoint> kps;
	std::cout << mkps.size() << '\n';
	for(auto& x : mkps) {
		cv::line(frame, x.cur.pt, x.pre.pt, cv::Scalar(255,0,0));
		kps.emplace_back(x.cur);
	}
	cv::drawKeypoints(frame, kps, frame);
	cv::imshow("Frame", frame);


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

