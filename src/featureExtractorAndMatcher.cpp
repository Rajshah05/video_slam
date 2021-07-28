#include<opencv2/opencv.hpp>
#include<iostream>
#include<Eigen/Dense>
#include"featureExtractorAndMatcher.h"


FeatureExtractorAndMatcher::FeatureExtractorAndMatcher() {
    orb = cv::ORB::create();
    // matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    // mK << F, 0, W/2, 0, F, H/2, 0, 0, 1;
    matcher = cv::BFMatcher(cv::NORM_HAMMING);
    // mKinv = K.inverse();

}

// cv::Mat FeatureExtractorAndMatcher::normalize(cv::Mat& pts) {
//     Eigen::Map<MatrixXf> pts_eigen( pts.data() ); 
//     pts_eigen.conservativeResize(NoChange, pts_eigen.cols()+1);
//     pts_eigen.col(pts_eigen.cols()-1) = Eigen::MatrixXf::Ones(pts_eigen.rows(),1);
//     pts_eigen = mKinv*pts_eigen.transpose();
//     pts_eigen = pts_eigen.transpose().block(0,0,pts_eigen.rows()-1,1);
//     cv::Mat ret(pts_eigen.rows(), pts_eigen.cols(), CV_32F, pts_eigen.data());
//     return ret;
// }

// cv::Mat FeatureExtractorAndMatcher::denormalize(cv::Mat& pt) {
//     Eigen::Map<MatrixXf> pt_eigen( pt.data() ); 
//     pt_eigen.conservativeResize(NoChange, pt_eigen.cols()+1);
//     pt_eigen.col(pt_eigen.cols()-1) = Eigen::MatrixXf::Ones(pt_eigen.rows(),1);
//     pt_eigen = mK*pt_eigen.transpose();
//     pt_eigen = pt_eigen.transpose().block(0,0,pt_eigen.rows()-1,1);
//     cv::Mat ret(pt_eigen.rows(), pt_eigen.cols(), CV_32F, pt_eigen.data());
//     return ret;
// }

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
    std::vector<match_kp> ret;
    // cv::Mat retMat(ret.size(), 4, CV_32F);
    Eigen::MatrixXf retMat(ret.size(), 4);

    if(!last_des.empty()) {
        std::vector< std::vector<cv::DMatch> > knn_matches;
        matcher.knnMatch( des, last_des, knn_matches, 2 );
        for (size_t i = 0; i < knn_matches.size(); i++){
            if (knn_matches[i][0].distance < 0.75f * knn_matches[i][1].distance)//.53
            {
                // match_kp temp = {kps[knn_matches[i][0].queryIdx], last_kps[knn_matches[i][0].trainIdx]};
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

        // p1 = normalize(p1);
        // p2 = normalize(p2);

        std::vector<uchar> mask(match_kps.size());
        cv::Mat F = cv::findFundamentalMat(p1, p2, cv::FM_RANSAC, 1, 0.99, 100, mask);
        for(int i = 0; i < match_kps.size(); i++) {
            if(mask[i]){
                ret.emplace_back(match_kps[i]);
            }
        }
        // p1.resize(ret.size());
        // p2.resize(ret.size());
        retMat.resize(ret.size(),4);
        for(int i = 0; i < ret.size(); i++) {
            retMat(i,0) = ret[i].cur.pt.x;
            retMat(i,1) = ret[i].cur.pt.y;
            retMat(i,2) = ret[i].pre.pt.x;
            retMat(i,3) = ret[i].pre.pt.y;
        }
        // std::cout << match_kps.size() << " " << ret.size() << '\n';
    }
    last_des = des;
    last_kps = kps;
    // match_kp_mat mkp = {p1, p2};
    return retMat;
    // return match_kps;
}
