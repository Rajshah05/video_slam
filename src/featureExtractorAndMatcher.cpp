#include<opencv2/opencv.hpp>
#include<iostream>
#include<Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include"featureExtractorAndMatcher.h"


FeatureExtractorAndMatcher::FeatureExtractorAndMatcher(Eigen::Matrix3f K) {
    orb = cv::ORB::create();
    mK = K;
    matcher = cv::BFMatcher(cv::NORM_HAMMING);
    mKinv = K.inverse();

}

cv::Mat FeatureExtractorAndMatcher::normalize(cv::Mat pts) {
    Eigen::MatrixXf pts_eigen;
    cv::cv2eigen(pts, pts_eigen);
    pts_eigen.conservativeResize(pts_eigen.rows(), pts_eigen.cols()+1);
    pts_eigen.col(pts_eigen.cols()-1) = Eigen::MatrixXf::Ones(pts_eigen.rows(),1);
    Eigen::MatrixXf pts_eigen_norm = (mKinv*(pts_eigen.transpose())).transpose().leftCols(2);
    cv::eigen2cv(pts_eigen_norm, pts);
    return pts;
}

Eigen::Vector2f FeatureExtractorAndMatcher::denormalize(Eigen::Vector3f pt) {
    Eigen::Vector3f ret;
    ret = mK*pt;
    ret(0) = std::round(ret(0));
    ret(1) = std::round(ret(1));
    return ret.head(2);
}

Eigen::MatrixXf FeatureExtractorAndMatcher::ExtractAndMatch(cv::Mat& frame) {
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
    // std::vector<match_kp> ret;
    std::vector<int> ret;
    // cv::Mat retMat(ret.size(), 4, CV_32F);
    Eigen::MatrixXf retMat(0, 4);

    if(!last_des.empty()) {
        std::vector< std::vector<cv::DMatch> > knn_matches;
        matcher.knnMatch( des, last_des, knn_matches, 2 );
        for (size_t i = 0; i < knn_matches.size(); i++){
            if (knn_matches[i][0].distance < 0.75f * knn_matches[i][1].distance)//.53
            {
                match_kps.emplace_back(match_kp{kps[knn_matches[i][0].queryIdx], last_kps[knn_matches[i][0].trainIdx]});
            }
        }

        //filter
        cv::Mat p1(match_kps.size(), 2, CV_32F);
        cv::Mat p2(match_kps.size(), 2, CV_32F);
        for(int i = 0; i < match_kps.size(); i++) {
            p1.at<float>(i,0) = match_kps[i].cur.pt.x;
            p1.at<float>(i,1) = match_kps[i].cur.pt.y;
            p2.at<float>(i,0) = match_kps[i].pre.pt.x;
            p2.at<float>(i,1) = match_kps[i].pre.pt.y;

        }
        cv::Mat p1n = normalize(p1);
        cv::Mat p2n = normalize(p2);

        std::vector<uchar> mask(match_kps.size());
        cv::Mat F = cv::findFundamentalMat(p1n, p2n, cv::FM_RANSAC, 1, 0.99, 100, mask);
        for(int i = 0; i < match_kps.size(); i++) {
            if(mask[i]){
                ret.emplace_back(i);
            }
        }
        retMat.resize(ret.size(),4);
        for(int i = 0; i < ret.size(); i++) {
            retMat(i,0) = p1n.at<float>(ret[i],0);
            retMat(i,1) = p1n.at<float>(ret[i],1);
            retMat(i,2) = p2n.at<float>(ret[i],0);
            retMat(i,3) = p2n.at<float>(ret[i],1);
        }
        std::cout << p1n.size() << " " << retMat.rows() << '\n';
    }
    last_des = des;
    last_kps = kps;
    return retMat;
}
