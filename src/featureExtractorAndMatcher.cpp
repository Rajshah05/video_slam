#include<opencv2/opencv.hpp>
#include<iostream>
#include"featureExtractorAndMatcher.h"
#include"frame_output.h"


FeatureExtractorAndMatcher::FeatureExtractorAndMatcher() {
    orb = cv::ORB::create();
    matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
}

Frame_output FeatureExtractorAndMatcher::ExtractAndMatch(cv::Mat& frame) {
    Frame_output fo;
    cv::Mat frame_grey, descriptors;
    
    // grey scaling frame	
    cv::cvtColor(frame, frame_grey,cv::COLOR_BGR2GRAY);

    // keypoint detection
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(frame_grey, corners, 3000, 0.01, 3);
    
    // std::vector<cv::KeyPoint> kps;
    fo.kps.reserve(corners.size());
    for(auto& x : corners) {
        fo.kps.emplace_back(x, 20);
    }


    // descriptor extraction
    orb->compute(frame, fo.kps, fo.des);

    if(!last_des.empty()) {
        std::vector< std::vector<cv::DMatch> > knn_matches; 
        fo.des.convertTo(fo.des, CV_32F);
        last_des.convertTo(last_des, CV_32F);
        matcher->knnMatch( fo.des, last_des, knn_matches, 2 );
        const float ratio_thresh = 0.7f;
        // std::cout << knn_matches.size() << '\n';
        for (size_t i = 0; i < knn_matches.size(); i++){
            if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
            {
                fo.knn_matches.emplace_back(knn_matches[i][0]);
            }
        }
    }
    last_des = fo.des;
    return fo;
}