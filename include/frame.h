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
	cv::Mat ptspts;
	cv::Mat Rt;
};


class Frame {
	public:
		Frame(const cv::Mat&, const cv::Mat&);
		// kps_des ExtractAndMatch(const cv::Mat&);
		// cv::Mat normalize(const cv::Mat&);
		// cv::Mat denormalize(const cv::Mat&);
		// cv::Mat ExtractRt(const cv::Mat&);
		// Eigen::MatrixXf ExtractRt(const cv::Mat&);

		cv::Mat mdes, mpts, mK, mKinv;
	private:
		// cv::Mat mK, mKinv;
		// std::vector<cv::Point2f> mpts;

		// cv::Ptr<cv::Feature2D> orb;
		// // cv::Ptr<cv::DescriptorMatcher> matcher;
		// cv::BFMatcher matcher;
		// cv::Mat last_des;
		// std::vector<cv::KeyPoint> last_kps;
};

pts_des extract(const cv::Mat&);
cv::Mat extractRt(const cv::Mat&);
cv::Mat normalize(const cv::Mat&, cv::Mat);
cv::Mat denormalize(const cv::Mat&, const cv::Mat&);
ptsptsRt matchAndRt(const Frame&, const Frame&);
