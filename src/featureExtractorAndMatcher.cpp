#include<opencv2/opencv.hpp>
#include<iostream>
#include<Eigen/Geometry>
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
    std::vector<match_kp> ret;

    if(!last_des.empty()) {
        std::vector< std::vector<cv::DMatch> > knn_matches; 
        // des.convertTo(des, CV_32F);
        // last_des.convertTo(last_des, CV_32F);
        // matcher->knnMatch( des, last_des, knn_matches, 2 );
        matcher.knnMatch( des, last_des, knn_matches, 2 );
        // std::cout << knn_matches.size() << '\n';
        // cv::Mat p1,p2;
        // std::cout << "bobo\n";
        for (size_t i = 0; i < knn_matches.size(); i++){
            if (knn_matches[i][0].distance < 0.75f * knn_matches[i][1].distance)//.53
            {
                // match_kp mkp = {}
                match_kp temp = {kps[knn_matches[i][0].queryIdx], last_kps[knn_matches[i][0].trainIdx]};
                match_kps.emplace_back(temp);
                // p1.push_back(cv::Mat({1,2},{temp.cur.pt.x, temp.cur.pt.y}));
                // p2.push_back(cv::Mat({1,2},{temp.pre.pt.x, temp.pre.pt.y})); 
            }
        }
        cv::Mat p1(match_kps.size(), 2, CV_32F);
        cv::Mat p2(match_kps.size(), 2, CV_32F);
        for(int i = 0; i < match_kps.size(); i++) {
            p1.at<float>(i,0) = match_kps[i].cur.pt.x;
            p1.at<float>(i,1) = match_kps[i].cur.pt.y;
            p2.at<float>(i,0) = match_kps[i].pre.pt.x;
            p2.at<float>(i,1) = match_kps[i].pre.pt.y;
            // std::cout << match_kps[i].cur.pt.x << " " << match_kps[i].cur.pt.y << "  " << p1.at<float>(i,0) << " " << p1.at<float>(i,1) << '\n';

        }
        // std::cin.get();
        // std::cout << p1.size() << "  " << p2.size();
        std::vector<uchar> mask(match_kps.size());
        // cv::Mat mask;
        // std::cout << match_kps.size() << " ";
        cv::Mat F = cv::findFundamentalMat(p1, p2, cv::FM_RANSAC, 3, 0.99, 100, mask);
        // std::cout << "bobo\n";
        // cv::Mat F = cv::findFundamentalMat(p1, p2, mask, cv::FM_RANSAC);
        for(int i = 0; i < match_kps.size(); i++) {
            if(mask[i]) ret.emplace_back(match_kps[i]);
            // if(mask.at<float>(i)!=0) ret.emplace_back(match_kps[i]);
        }
        std::cout << match_kps.size() << " " << ret.size() << '\n'; 
    }
    last_des = des;
    last_kps = kps;
    return ret;
    // return match_kps;
}
