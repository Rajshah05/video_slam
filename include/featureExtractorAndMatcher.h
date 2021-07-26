#pragma once
#include<opencv2/opencv.hpp>
#include<iostream>
#include"frame_output.h"

class FeatureExtractorAndMatcher {
	public:
		FeatureExtractorAndMatcher();
		Frame_output ExtractAndMatch(cv::Mat& frame);



	private:
		cv::Ptr<cv::Feature2D> orb;
		cv::Ptr<cv::DescriptorMatcher> matcher;
		cv::Mat last_des;
};
