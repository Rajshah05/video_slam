#include"frame.h"
#include<iostream>
#include<opencv2/opencv.hpp>
#include<Eigen/Dense>
#include<complex>
#include<math.h>
#include<vector>

#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/factory.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
// #include "g2o/solvers/csparse/linear_solver_csparse.h"
 
#include "g2o/types/slam2d/vertex_se2.h"
#include "g2o/types/slam3d/vertex_se3.h"



static const int W = 1920/2;
static const int H = 1080/2;
static const int F = 270;//270






void process_frame(std::vector<Frame> & frames, cv::Mat& frame, const cv::Mat& K) {

	frames.emplace_back(Frame(frame, K));
	if(frames.size() <= 1) return;

	auto [matchCoords, Rt] = matchAndRt(frames[frames.size()-1], frames[frames.size()-2]);
	
	if(!matchCoords.rows) return;

	cv::Mat curv(3,1,CV_32F);
	cv::Mat prev(3,1,CV_32F);
	for(int i = 0; i < matchCoords.rows; i++) {
		curv.at<float>(0,0) = matchCoords.at<float>(i,0);
		curv.at<float>(1,0) = matchCoords.at<float>(i,1);
		curv.at<float>(2,0) = 1.0;
		prev.at<float>(0,0) = matchCoords.at<float>(i,2);
		prev.at<float>(1,0) = matchCoords.at<float>(i,3);
		prev.at<float>(2,0) = 1.0;
		cv::Mat curi = denormalize(K, curv);
		cv::Mat prei = denormalize(K, prev);
		cv::line(frame, cv::Point(curi.at<float>(0,0), curi.at<float>(1,0)), cv::Point(prei.at<float>(0,0), prei.at<float>(1,0)), cv::Scalar(255,0,0));
		cv::circle(frame, cv::Point(curi.at<float>(0,0), curi.at<float>(1,0)), 3, cv::Scalar(0,255,0));
	}
	cv::imshow("Frame", frame);


}


int main(int argc, char** argv) {

	cv::VideoCapture cap("../video/test_countryroad.mp4");	
	if(!cap.isOpened()) {
		std::cout << "Error opening video stream or file" << '\n';
		return -1;
	}
	cv::Mat K = (cv::Mat_<float>(3,3) << float(F), 0, float(W/2), 0, float(F), float(H/2), 0, 0, 1);
	std::vector<Frame> frames;
	
	while(1) {
		cv::Mat frame, frame_rsz;
		cap >> frame;
		
		// Resizing frame
		cv::resize(frame, frame_rsz, cv::Size(W,H));
		
		if(frame.empty()) {
			break;
		}
		process_frame(frames, frame_rsz, K);

		char c = (char)cv::waitKey(25);
		if(c == 27) break;
	}
	cap.release();
	cv::destroyAllWindows();
	return 0;
}

