#pragma once
#include<opencv2/opencv.hpp>
#include<iostream>
#include"frame_output.h"

struct match_kp {
	cv::KeyPoint cur,pre;
};

class FeatureExtractorAndMatcher {
	public:
		FeatureExtractorAndMatcher();
		std::vector<match_kp> ExtractAndMatch(cv::Mat& frame);



	private:
		cv::Ptr<cv::Feature2D> orb;
		// cv::Ptr<cv::DescriptorMatcher> matcher;
		cv::BFMatcher matcher;
		cv::Mat last_des;
		std::vector<cv::KeyPoint> last_kps;
};
