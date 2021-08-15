#include<opencv2/opencv.hpp>
#include<iostream>
#include<Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include"frame.h"


Frame::Frame(const cv::Mat& frame, const cv::Mat& K) {
    mK = K;
    mKinv = mK.inv();
    auto [pts, des] = extract(frame);
    // mpts = normalize(mKinv, pts);
    mpts = pts;
    mdes=des;
}

cv::Mat extractRt(const cv::Mat& E) {
    cv::Mat d, U, Vt;
    // cv::SVDecomp(E, d, U, Vt, cv::SVD::FULL_UV);
    cv::SVD::compute(E,d,U,Vt);
    std::cout << cv::determinant(U) << " " << cv::determinant(Vt) << '\n';
    if (cv::determinant(U) < 0) {
        U.col(2) *= -1;
    }

    // Last row of Vt is undetermined since d = (a a 0).
    if (cv::determinant(Vt) < 0) {
        Vt.row(2) *= -1;
    }

    cv::Mat W = (cv::Mat_<double>(3, 3) << 0, -1, 0,
        1, 0, 0,
        0, 0, 1);

    cv::Mat Rt = cv::Mat::eye(4,4,CV_32F);
    cv::Mat R;
    R = (U*W)*Vt;
    
    if(cv::sum(R.diag(0))[0] < 0) {
        R = (U*W.t())*Vt;
    }
    cv::Mat t = U.col(2);

    cv::hconcat(R,t,R);

    R.copyTo(Rt(cv::Rect(0,0,4,3)));
    // std::cout << Rt << '\n';
    return Rt;
}

cv::Mat normalize(const cv::Mat& Kinv, const cv::Mat& pts) {
    cv::Mat ret;
    cv::hconcat(pts,cv::Mat::ones(pts.rows,1,CV_32F),ret);
    ret = (Kinv*ret.t()).t();
    return ret(cv::Rect(0,0,2,pts.rows));
}

cv::Mat denormalize(const cv::Mat& K, const cv::Mat& pt) {
    cv::Mat ret;
    ret = K*pt;
    ret.at<float>(0,0) = std::round(ret.at<float>(0,0)/ret.at<float>(2,0));
    ret.at<float>(1,0) = std::round(ret.at<float>(1,0)/ret.at<float>(2,0));
    ret.resize(2);
    return ret;
}

pts_des extract(const cv::Mat& frame) {
    cv::Ptr<cv::Feature2D> orb{cv::ORB::create()};
    
    // grey scaling frame
    cv::Mat frame_grey;
    cv::cvtColor(frame, frame_grey,cv::COLOR_BGR2GRAY);

    // keypoint detection
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(frame_grey, corners, 5000, 0.01, 3);

    cv::Mat pts(corners.size(), 2, CV_32F);
    std::vector<cv::KeyPoint> kps;
    kps.reserve(corners.size());
    for(int i = 0; i < corners.size(); ++i) {
        kps.emplace_back(corners[i], 20);
    }

    // descriptor extraction using kps
    cv::Mat des;
    orb->compute(frame, kps, des); // MODIFIES KPS

    for(int i = 0; i < kps.size(); ++i) {
        pts.at<float>(i,0) = kps[i].pt.x;
        pts.at<float>(i,1) = kps[i].pt.y;
    }

    // return pts and des   
    return {pts,des};
}


ptsptsRt matchAndRt(const Frame& f1, const Frame& f2) {
    cv::BFMatcher matcher{cv::BFMatcher(cv::NORM_HAMMING)};
    cv::Mat filteredKPmat(0, 4, CV_32F);

    std::vector< std::vector<cv::DMatch> > knn_matches;

    matcher.knnMatch( f1.mdes, f2.mdes, knn_matches, 2 );

    cv::Mat p1(knn_matches.size(), 2, CV_32F);
    cv::Mat p2(knn_matches.size(), 2, CV_32F);
    int cur_row = 0;
    for (size_t i = 0; i < knn_matches.size(); ++i){
        if (knn_matches[i][0].distance < 0.75f * knn_matches[i][1].distance)//.53
        {
            p1.at<float>(cur_row,0) = f1.mpts.at<float>(knn_matches[i][0].queryIdx,0);
            p1.at<float>(cur_row,1) = f1.mpts.at<float>(knn_matches[i][0].queryIdx,1);
            p2.at<float>(cur_row,0) = f2.mpts.at<float>(knn_matches[i][0].trainIdx,0);
            p2.at<float>(cur_row,1) = f2.mpts.at<float>(knn_matches[i][0].trainIdx,1);

            cur_row++;
        }
    }
    p1.resize(cur_row);
    p2.resize(cur_row);

    //filter
    
    filteredKPmat.resize(cur_row);
    std::vector<uchar> mask(cur_row);
    // std::cout << p1.at<float>(0,0) << " " << p1.at<float>(0,1) << " " << p2.at<float>(0,0) << " " << " " << p2.at<float>(0,1) << '\n';
    // const cv::Mat E = cv::findEssentialMat(p1, p2, f1.mK, cv::RANSAC, 0.999, 0.005, 200, mask);
    // std::cout << f1.mKinv << '\n';
    const cv::Mat E = cv::findEssentialMat(p1, p2, f1.mK.at<float>(0,0), cv::Point2f(f1.mK.at<float>(0,2),f1.mK.at<float>(1,2)), cv::RANSAC, 0.999, 1.0, mask);
    
    cur_row = 0;
    for(int i = 0; i < p1.rows; i++) {
        if(mask[i]){
            filteredKPmat.at<float>(cur_row,0) = p1.at<float>(i,0);
            filteredKPmat.at<float>(cur_row,1) = p1.at<float>(i,1);
            filteredKPmat.at<float>(cur_row,2) = p2.at<float>(i,0);
            filteredKPmat.at<float>(cur_row,3) = p2.at<float>(i,1);

            cur_row++;
        }
    }

    filteredKPmat.resize(cur_row);
    // std::cout << mask.size() << " " << filteredKPmat.rows << '\n';
    // std::cout << E << '\n';
    // std::cin.get();
    // cv::Mat Rt = extractRt(E);
   
    cv::Mat R,t;
    cv::Mat Rt = cv::Mat::eye(4,4,CV_32F);
    // cv::recoverPose(E,p1,p2,f1.mK,R,t,7);
    mask.clear();
    // cv::recoverPose(E, p1, p2, R, t, f1.mK.at<float>(0,0), cv::Point2f(f1.mK.at<float>(0,2),f1.mK.at<float>(1,2)), mask);
    cv::recoverPose(E, filteredKPmat(cv::Rect(0,0,2,cur_row)),filteredKPmat(cv::Rect(2,0,2,cur_row)) , R, t, f1.mK.at<float>(0,0), cv::Point2f(f1.mK.at<float>(0,2),f1.mK.at<float>(1,2)), mask);
    cv::hconcat(R,t,R);
    R.copyTo(Rt(cv::Rect(0,0,4,3)));
   
    cur_row = 0;
    for(int i = 0; i < filteredKPmat.rows; ++i) {
        if(mask[i]) {
            filteredKPmat.at<float>(cur_row,0) = filteredKPmat.at<float>(i,0);
            filteredKPmat.at<float>(cur_row,1) = filteredKPmat.at<float>(i,1);
            filteredKPmat.at<float>(cur_row,2) = filteredKPmat.at<float>(i,2);
            filteredKPmat.at<float>(cur_row,3) = filteredKPmat.at<float>(i,3);
            cur_row++;
        }
    }

    // std::cout << Rt << '\n';
    // std::cin.get();
    
    return {filteredKPmat, Rt};
}
