#include<opencv2/opencv.hpp>
#include<iostream>
#include<Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include"featureExtractorAndMatcher.h"


FeatureExtractorAndMatcher::FeatureExtractorAndMatcher(cv::Mat K) {
    orb = cv::ORB::create();
    mK = K;
    matcher = cv::BFMatcher(cv::NORM_HAMMING);
    mKinv = K.inv();

}

// Eigen::MatrixXf FeatureExtractorAndMatcher::ExtractRt(cv::Mat& E) {
//     Eigen::MatrixXf Ee;
//     cv2eigen(E, Ee);
//     Eigen::Matrix3f W;
//     W << 0,-1,0,1,0,0,0,0,1;
//     JacobiSVD<MatrixXf> svd( E, ComputeThinU | ComputeThinV);
//     assert(svd.matrixU().determinant() > 0);
//     if (svd.matrixV().determinant() < 0) {
//         svd.matrixV()*=-1;
//     }
//     Eigen::Matrix3f R = (svd.matrixU()*W)*svd.matrixV();
// }

cv::Mat FeatureExtractorAndMatcher::normalize(cv::Mat& pts) {

    Eigen::MatrixXf pts_eigen, mKinv_e;
    cv::cv2eigen(pts, pts_eigen);
    cv::cv2eigen(mKinv, mKinv_e);
    pts_eigen.conservativeResize(pts_eigen.rows(), pts_eigen.cols()+1);
    pts_eigen.col(pts_eigen.cols()-1) = Eigen::MatrixXf::Ones(pts_eigen.rows(),1);
    Eigen::MatrixXf pts_eigen_norm = (mKinv_e*(pts_eigen.transpose())).transpose().leftCols(2);
    cv::eigen2cv(pts_eigen_norm, pts);
    return pts;
}

cv::Mat FeatureExtractorAndMatcher::denormalize(cv::Mat& pt) {
    cv::Mat ret;
    ret = mK*pt;
    ret.at<float>(0,0) = std::round(ret.at<float>(0,0));
    ret.at<float>(1,0) = std::round(ret.at<float>(1,0));
    ret.resize(2);
    return ret;
}

cv::Mat FeatureExtractorAndMatcher::ExtractAndMatch(cv::Mat& frame) {
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

    std::vector<int> ret;
    cv::Mat filteredKPmat(0, 4, CV_32F);
    if(!last_des.empty()) {
        std::vector< std::vector<cv::DMatch> > knn_matches;
        matcher.knnMatch( des, last_des, knn_matches, 2 );
        cv::Mat p1(knn_matches.size(), 2, CV_32F);
        cv::Mat p2(knn_matches.size(), 2, CV_32F);
        int cur_row = 0;
        for (size_t i = 0; i < knn_matches.size(); i++){
            if (knn_matches[i][0].distance < 0.75f * knn_matches[i][1].distance)//.53
            {
                p1.at<float>(cur_row,0) = kps[knn_matches[i][0].queryIdx].pt.x;
                p1.at<float>(cur_row,1) = kps[knn_matches[i][0].queryIdx].pt.y;
                p2.at<float>(cur_row,0) = last_kps[knn_matches[i][0].trainIdx].pt.x;
                p2.at<float>(cur_row,1) = last_kps[knn_matches[i][0].trainIdx].pt.y;

                cur_row++;
            }
        }
        p1.resize(cur_row);
        p2.resize(cur_row);
        //filter
        
        cv::Mat p1n = normalize(p1);
        cv::Mat p2n = normalize(p2);
        std::vector<uchar> mask(cur_row);
        cur_row = 0;
        cv::Mat E = cv::findEssentialMat(p1n, p2n, mK, cv::RANSAC, 0.99, 0.005, 100, mask);
        for(int i = 0; i < p1n.rows; i++) {
            if(mask[i]){
                p1.at<float>(cur_row,0) = p1n.at<float>(i,0);
                p1.at<float>(cur_row,1) = p1n.at<float>(i,1);
                p2.at<float>(cur_row,0) = p2n.at<float>(i,0);
                p2.at<float>(cur_row,1) = p2n.at<float>(i,1);
                cur_row++;
            }
        }

        p1.resize(cur_row);
        p2.resize(cur_row);

        cv::hconcat(p1(cv::Rect(0,0,2,p1.rows)), p2(cv::Rect(0,0,2,p2.rows)), filteredKPmat);

        std::cout << mask.size() << " " << filteredKPmat.rows << '\n';

        
        // Eigen::Matrix3f W;
        // W << 0,-1,0,1,0,0,0,0,1;
        // Eigen::MatrixXf Ee,U,V;
        // cv2eigen(E, Ee);
        // // std::cout << Ee << '\n';
        // Eigen::JacobiSVD<Eigen::MatrixXf> svd( Ee, Eigen::ComputeFullU | Eigen::ComputeFullV);
        // // std::cout << svd.matrixU().determinant() << " " << svd.matrixV().determinant() << '\n';
        // // std::cout << svd.singularValues() << '\n';
        // V = svd.matrixV();
        // U = svd.matrixU();
        // // assert(svd.matrixU().determinant() > 0);
        // // std::cout << U << " ";
        // if (U.determinant() < 0) {
        //     U*=-1;
        // }
        // // std::cout << U << '\n';
        // if (V.determinant() < 0) {
        //     V*=-1;
        // }
        // Eigen::Matrix3f R = (U*W)*V;
        // std::cout << R << '\n';
    }
    last_des = des;
    last_kps = kps;
    return filteredKPmat;
}
