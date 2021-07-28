#include"featureExtractorAndMatcher.h"
#include<iostream>
#include<opencv2/opencv.hpp>
#include<Eigen/Dense>
#include<complex>
#include<math.h>
#include<vector>


static const int W = 1920/2;
static const int H = 1080/2;
static const int F = 1;

// Eigen::Matrix2i abc;
// abc << 1,1,1,1;
// std::cout << "abs";
// std::cout << K;





void process_frame(FeatureExtractorAndMatcher* fem, cv::Mat& frame) {
	
	Eigen::MatrixXf mkps = fem->ExtractAndMatch(frame);

	if(!mkps.rows()) return;
	for(int i = 0; i < mkps.rows(); i++) {
		Eigen::Vector3f curv, prev;
		curv << mkps(i,0), mkps(i,1), 1.0;
		prev << mkps(i,2), mkps(i,3), 1.0;
		Eigen::Vector2f curi = fem->denormalize(curv);
		Eigen::Vector2f prei = fem->denormalize(prev);
		cv::line(frame, cv::Point(curi(0), curi(1)), cv::Point(prei(0), prei(1)), cv::Scalar(255,0,0));
		cv::circle(frame, cv::Point(curi(0), curi(1)), 3, cv::Scalar(0,255,0));
	}
	cv::imshow("Frame", frame);


}


int main(int argc, char** argv) {

	cv::VideoCapture cap("../video/test_countryroad.mp4");	
	if(!cap.isOpened()) {
		std::cout << "Error opening video stream or file" << '\n';
		return -1;
	}

	Eigen::Matrix3f K;
	K << float(F), 0, float(W/2), 0, float(F), float(H/2), 0, 0, 1;
	FeatureExtractorAndMatcher* fem = new FeatureExtractorAndMatcher(K);
	
	
	while(1) {
		cv::Mat frame, frame_rsz;
		// Frame_output frame_o;
		cap >> frame;
		
		// Resizing frame
		cv::resize(frame, frame_rsz, cv::Size(W,H));
		
		if(frame.empty()) {
			break;
		}

		process_frame(fem, frame_rsz);

		char c = (char)cv::waitKey(25);
		if(c == 27) break;
	}
	cap.release();
	cv::destroyAllWindows();
	return 0;
}

