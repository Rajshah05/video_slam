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

Map mapp;

// void display_map(const std::vector<Frame>& frames, const std::vector<Point>& points) {
// 	for(auto& x : frames) {
// 		std::cout << x.pose << '\n';
// 	}
// 	for(auto& x : points) {
// 		std::cout << x.mxyz << '\n';
// 	}
// }


void process_frame(std::vector<Frame> & frames, std::vector<Point>& points, cv::Mat& frame, const cv::Mat& K) {

	frames.emplace_back(Frame(frame, K));
	if(frames.size() <= 1) return;

	Frame f1 = frames[frames.size()-1];
	Frame f2 = frames[frames.size()-2];

	auto [ind1, ind2, matchCoords, Rt] = matchAndRt(f1, f2);
	// if(!matchCoords.rows) return;

	frames[frames.size()-1].pose = Rt*frames[frames.size()-2].pose;
	// std::cout << frames[frames.size()-1].pose << '\n';
	// std::cout << pts4d << '\n';	
	cv::Mat pts4d;
	cv::Mat Rt0 = cv::Mat::eye(3, 4, CV_32F);
	cv::triangulatePoints(frames[0].mK*Rt0, frames[0].mK*Rt(cv::Rect(0,0,4,3)), matchCoords(cv::Rect(0,0,2,matchCoords.rows)).t(), matchCoords(cv::Rect(2,0,2,matchCoords.rows)).t(), pts4d);
	// pts4d = pts4d.t();
	// std::cout << matchCoords.rows << " " << pts4d.cols << " ";
	cv::Mat good_pts4d(pts4d.cols, 3, CV_32F);
	int cr = 0; 
	for (int i = 0; i < pts4d.cols; ++i) {
		if(abs(pts4d.at<float>(3,i)) > 0.005 && pts4d.at<float>(2,i)/pts4d.at<float>(3,i) > 0) {
			good_pts4d.at<float>(cr,0) = pts4d.at<float>(0,i)/pts4d.at<float>(3,i);
			good_pts4d.at<float>(cr,1) = pts4d.at<float>(1,i)/pts4d.at<float>(3,i);
			good_pts4d.at<float>(cr,2) = pts4d.at<float>(2,i)/pts4d.at<float>(3,i);
			
			
			Point pt(cv::Point3f(good_pts4d.at<float>(cr,0), good_pts4d.at<float>(cr,1), good_pts4d.at<float>(cr,2)));
			pt.add_observation(frames[frames.size()-1], ind1.at<int>(i,0));
			pt.add_observation(frames[frames.size()-2], ind2.at<int>(i,0));
			points.emplace_back(pt);

			cr++;
		}
	}
	good_pts4d.resize(cr);
	// std::cout << good_pts4d.rows << '\n';



	cv::Mat curv(3,1,CV_32F);
	cv::Mat prev(3,1,CV_32F);
	for(int i = 0; i < matchCoords.rows; ++i) {
		curv.at<float>(0,0) = matchCoords.at<float>(i,0);
		curv.at<float>(1,0) = matchCoords.at<float>(i,1);
		curv.at<float>(2,0) = 1.0;
		prev.at<float>(0,0) = matchCoords.at<float>(i,2);
		prev.at<float>(1,0) = matchCoords.at<float>(i,3);
		prev.at<float>(2,0) = 1.0;
		// cv::Mat curi = denormalize(K, curv);
		// cv::Mat prei = denormalize(K, prev);
		// cv::line(frame, cv::Point(curi.at<float>(0,0), curi.at<float>(1,0)), cv::Point(prei.at<float>(0,0), prei.at<float>(1,0)), cv::Scalar(255,0,0));
		// cv::circle(frame, cv::Point(curi.at<float>(0,0), curi.at<float>(1,0)), 3, cv::Scalar(0,255,0));
		cv::line(frame, cv::Point(curv.at<float>(0,0), curv.at<float>(1,0)), cv::Point(prev.at<float>(0,0), prev.at<float>(1,0)), cv::Scalar(255,0,0));
		cv::circle(frame, cv::Point(curv.at<float>(0,0), curv.at<float>(1,0)), 3, cv::Scalar(0,255,0));
	}
	cv::imshow("Frame", frame);

	mapp.display();

}




int main(int argc, char** argv) {

	cv::VideoCapture cap("../video/test_countryroad.mp4");	
	if(!cap.isOpened()) {
		std::cout << "Error opening video stream or file" << '\n';
		return -1;
	}
	const cv::Mat K = (cv::Mat_<float>(3,3) << int(F), 0, int(W/2), 0, int(F), int(H/2), 0, 0, 1);
	std::vector<Frame> frames;
	std::vector<Point> points;
	
	while(1) {
		cv::Mat frame, frame_rsz;
		cap >> frame;
		
		// Resizing frame
		cv::resize(frame, frame_rsz, cv::Size(W,H));
		
		if(frame.empty()) {
			break;
		}
		process_frame(frames, points, frame_rsz, K);

		// display_map(frames, points);

		char c = (char)cv::waitKey(25);
		if(c == 27) break;
	}
	cap.release();
	cv::destroyAllWindows();
	return 0;
}

