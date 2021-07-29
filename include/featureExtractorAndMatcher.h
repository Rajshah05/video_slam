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

class FeatureExtractorAndMatcher {
	public:
		FeatureExtractorAndMatcher(cv::Mat);
		cv::Mat ExtractAndMatch(cv::Mat&);
		cv::Mat normalize(cv::Mat&);
		cv::Mat denormalize(cv::Mat&);
		// Eigen::MatrixXf ExtractRt(const cv::Mat&);


	private:
		cv::Ptr<cv::Feature2D> orb;
		// cv::Ptr<cv::DescriptorMatcher> matcher;
		cv::BFMatcher matcher;
		cv::Mat last_des;
		std::vector<cv::KeyPoint> last_kps;
		cv::Mat mK;
		cv::Mat mKinv;
};
