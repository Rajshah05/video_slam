#include<opencv2/opencv.hpp>
#include<iostream>
#include"featureExtractorAndMatcher.h"
#include"frame_output.h"


FeatureExtractorAndMatcher::FeatureExtractorAndMatcher() {
    orb = cv::ORB::create();
    // matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    matcher = cv::BFMatcher(cv::NORM_HAMMING);
}

std::vector<match_kp> FeatureExtractorAndMatcher::ExtractAndMatch(cv::Mat& frame) {
    // grey scaling frame	
    cv::Mat frame_grey;
    cv::cvtColor(frame, frame_grey,cv::COLOR_BGR2GRAY);

    // keypoint detection
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(frame_grey, corners, 5000, 0.01, 3);
    
    std::vector<cv::KeyPoint> kps;
    kps.reserve(corners.size());
    for(auto& x : corners) {
        kps.emplace_back(x, 20);
    }


    // descriptor extraction
    cv::Mat des;
    orb->compute(frame, kps, des);

    std::vector<match_kp> match_kps;

    if(!last_des.empty()) {
        std::vector< std::vector<cv::DMatch> > knn_matches; 
        // des.convertTo(des, CV_32F);
        // last_des.convertTo(last_des, CV_32F);
        // matcher->knnMatch( des, last_des, knn_matches, 2 );
        matcher.knnMatch( des, last_des, knn_matches, 2 );
        // std::cout << knn_matches.size() << '\n';
        for (size_t i = 0; i < knn_matches.size(); i++){
            if (knn_matches[i][0].distance < 0.75f * knn_matches[i][1].distance)
            {
                // match_kp mkp = {}
                match_kp temp = {kps[knn_matches[i][0].queryIdx], last_kps[knn_matches[i][0].trainIdx]};
                match_kps.emplace_back(temp);
            }
        }
    }
    last_des = des;
    last_kps = kps;
    return match_kps;
}