#include<opencv2/opencv.hpp>
#include<iostream>
#include<Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include"featureExtractorAndMatcher.h"


FeatureExtractorAndMatcher::FeatureExtractorAndMatcher(const cv::Mat& K) {
    orb = cv::ORB::create();
    mK = K;
    matcher = cv::BFMatcher(cv::NORM_HAMMING);
    mKinv = K.inv();

}

cv::Mat FeatureExtractorAndMatcher::ExtractRt(const cv::Mat& E) {
    cv::Mat d, U, Vt;
    cv::SVDecomp(E, d, U, Vt, cv::SVD::FULL_UV);
    

    // std::cout << d.size() << " " << U.size() << " " << Vt.size() << '\n';
    if (cv::determinant(U) < 0) {
        U *= -1;
    }

    // Last row of Vt is undetermined since d = (a a 0).
    if (cv::determinant(Vt) < 0) {
        Vt *= -1;
    }
    // std::cout << cv::determinant(U) << " " << cv::determinant(Vt) << '\n';

    cv::Mat W = (cv::Mat_<double>(3, 3) << 0, -1, 0,
        1, 0, 0,
        0, 0, 1);

    cv::Mat R;
    R = (U*W)*Vt;
    // std:: cout << cv::sum(R.diag(0))[0] << '\n';
    
    if(cv::sum(R.diag(0))[0] < 0) {
        // std::cout << "bbobo---------------\n";
        // cv::Mat Wt;
        // cv::transpose(W,Wt);
        R = (U*W.t())*Vt;
    }
    cv::Mat t = U.col(2);

    cv::Mat Rt;
    cv::hconcat(R,t,Rt);
    return Rt;
}

cv::Mat FeatureExtractorAndMatcher::normalize(const cv::Mat& pts) {

    Eigen::MatrixXf pts_eigen, mKinv_e;
    cv::cv2eigen(pts, pts_eigen);
    cv::cv2eigen(mKinv, mKinv_e);
    pts_eigen.conservativeResize(pts_eigen.rows(), pts_eigen.cols()+1);
    pts_eigen.col(pts_eigen.cols()-1) = Eigen::MatrixXf::Ones(pts_eigen.rows(),1);
    Eigen::MatrixXf pts_eigen_norm = (mKinv_e*(pts_eigen.transpose())).transpose().leftCols(2);
    cv::eigen2cv(pts_eigen_norm, pts);
    return pts;
}

cv::Mat FeatureExtractorAndMatcher::denormalize(const cv::Mat& pt) {
    cv::Mat ret;
    ret = mK*pt;
    ret.at<float>(0,0) = std::round(ret.at<float>(0,0));
    ret.at<float>(1,0) = std::round(ret.at<float>(1,0));
    ret.resize(2);
    return ret;
}

cv::Mat FeatureExtractorAndMatcher::ExtractAndMatch(const cv::Mat& frame) {
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
        const cv::Mat E = cv::findEssentialMat(p1n, p2n, mK, cv::RANSAC, 0.99, 0.005, 100, mask);
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

        cv::Mat Rt = ExtractRt(E);
        // std::cout << mask.size() << " " << filteredKPmat.rows << '\n';
        
        std::cout << Rt << '\n';
    }
    last_des = des;
    last_kps = kps;
    return filteredKPmat;
}
