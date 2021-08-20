#pragma once
#include<opencv2/opencv.hpp>
#include<iostream>
#include<Eigen/Dense>

extern Eigen::Matrix3f K;

struct match_kp {
	cv::KeyPoint cur,pre;
};

struct match_kp_mat {
	cv::Mat cur, pre;
};

struct pts_des {
	// std::vector<cv::Point2f> pts;
	cv::Mat pts;
	cv::Mat des;
};

struct ptsptsRt {
	cv::Mat idx1;
	cv::Mat idx2;
	cv::Mat ptspts;
	cv::Mat Rt;
};

inline cv::Mat IRt = cv::Mat::eye(4,4,CV_32F);

class Frame {
	public:
		Frame(const cv::Mat&, const cv::Mat&);
		cv::Mat mdes, mpts, mK, mKinv, pose = IRt;
		int id;
	private:
};

pts_des extract(const cv::Mat&);
cv::Mat extractRt(const cv::Mat&);
cv::Mat normalize(const cv::Mat&, const cv::Mat&);
cv::Mat denormalize(const cv::Mat&, const cv::Mat&);
ptsptsRt matchAndRt(const Frame&, const Frame&);


class Point {
	public:
		Point(const cv::Point3f& xyz);
		void add_observation(const Frame& frame, const int& idx);

	public:
		cv::Point3f mxyz;
		std::vector<Frame> mframes;
		std::vector<int> midxs;
};

class Map {
	public:
		void display();

	public:
		std::vector<Frame> frames;
		std::vector<Point> points;
};