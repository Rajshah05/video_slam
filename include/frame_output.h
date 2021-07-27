#pragma once
#include<opencv2/opencv.hpp>
#include<iostream>

struct Frame_output {
    std::vector<cv::KeyPoint> kps;
    cv::Mat des;
    std::vector<cv::DMatch> matches;
};