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
		FeatureExtractorAndMatcher(Eigen::Matrix3f);
		Eigen::MatrixXf ExtractAndMatch(cv::Mat&);
		cv::Mat normalize(cv::Mat);
		Eigen::Vector2f denormalize(Eigen::Vector3f);


	private:
		cv::Ptr<cv::Feature2D> orb;
		// cv::Ptr<cv::DescriptorMatcher> matcher;
		cv::BFMatcher matcher;
		cv::Mat last_des;
		std::vector<cv::KeyPoint> last_kps;
		Eigen::Matrix3f mK;
		Eigen::Matrix3f mKinv;
};
