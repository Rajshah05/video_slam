#include<opencv2/opencv.hpp>
#include<iostream>
#include<Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include"frame.h"


Frame::Frame(const cv::Mat& frame, const cv::Mat& K) {
    // orb = cv::ORB::create();
    mK = K;
    mKinv = mK.inv();
    auto [pts, des] = extract(frame);
    // std::cout << pts.at<float>(0,0) << " " << pts.at<float>(0,1) << '\n';
    mpts = normalize(mKinv, pts);
    // std::cout << mpts.rows;
    // std::cout << mpts.at<float>(1,0) << " " << mpts.at<float>(1,1) << '\n';
    // std::cin.get();
    mdes=des;
}

cv::Mat extractRt(const cv::Mat& E) {
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

cv::Mat normalize(const cv::Mat& Kinv, const cv::Mat& pts) {
    // std::cout << pts(cv::Rect(0,5,2,10))<<'\n';
    cv::Mat ret;
    cv::hconcat(pts,cv::Mat::ones(pts.rows,1,CV_32F),ret);
    ret = (Kinv*ret.t()).t();
    // std::cout << ret(cv::Rect(0,5,2,10));
    // std::cin.get();
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
    // std::vector<cv::Point2f> pts_vec;
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(frame_grey, corners, 5000, 0.01, 3);

    // getting kps from pts
    cv::Mat pts(corners.size(), 2, CV_32F);
    std::vector<cv::KeyPoint> kps;
    kps.reserve(corners.size());
    for(int i = 0; i < corners.size(); ++i) {
        kps.emplace_back(corners[i], 20);
        pts.at<float>(i,0) = corners[i].x;
        pts.at<float>(i,1) = corners[i].y;
    }
    // for(int i = 0; i < corners.rows; ++i) {
    //     kps.emplace_back(pts_point2f.at<cv::Point2f>(i,0), 20);
    //     pts.at<float>(i,0) = pts_point2f.at<cv::Point2f>(i,0).x;
    //     pts.at<float>(i,1) = pts_point2f.at<cv::Point2f>(i,0).y;
    // }

    // descriptor extraction using kps
    cv::Mat des;
    // std::cout << kps[0].pt.x << " " << kps[0].pt.y << '\n';
    // std::cin.get();
    orb->compute(frame, kps, des);
    // std::cout << pts(cv::Rect(0,0,2,5)) << '\n';
    // std::cout << des(cv::Rect(0,0,2,5)) << '\n';
    // std::cin.get();
    // return pts and des   
    return {pts,des};
}


ptsptsRt matchAndRt(const Frame& f1, const Frame& f2) {
    cv::BFMatcher matcher{cv::BFMatcher(cv::NORM_HAMMING)};
    // std::vector<int> ret;
    cv::Mat filteredKPmat(0, 4, CV_32F);
    // if(!last_des.empty()) {

    std::vector< std::vector<cv::DMatch> > knn_matches;

    // std::cout << f2.mpts(cv::Rect(0,0,2,5)) << '\n';
    // std::cin.get();

    matcher.knnMatch( f1.mdes, f2.mdes, knn_matches, 2 );
    // std::cout << knn_matches.size() << '\n';
    cv::Mat p1(knn_matches.size(), 2, CV_32F);
    cv::Mat p2(knn_matches.size(), 2, CV_32F);
    int cur_row = 0;
    for (size_t i = 0; i < knn_matches.size(); ++i){
        if (knn_matches[i][0].distance < 0.75f * knn_matches[i][1].distance)//.53
        {
            // std::cout << knn_matches[i][0].trainIdx << '\n';
            // std::cin.get();
            p1.at<float>(cur_row,0) = f1.mpts.at<float>(knn_matches[i][0].queryIdx,0);
            p1.at<float>(cur_row,1) = f1.mpts.at<float>(knn_matches[i][0].queryIdx,1);
            p2.at<float>(cur_row,0) = f2.mpts.at<float>(knn_matches[i][0].trainIdx,0);
            p2.at<float>(cur_row,1) = f2.mpts.at<float>(knn_matches[i][0].trainIdx,1);

            cur_row++;
        }
    }
    // std::cout << cur_row << '\n';
    p1.resize(cur_row);
    p2.resize(cur_row);

    // std::cout << p1.cols << '\n';
    //filter
    
    // cv::Mat p1n = normalize(p1);
    // cv::Mat p2n = normalize(p2);
    filteredKPmat.resize(cur_row);
    std::vector<uchar> mask(cur_row);

    // std::cout << p1.at<float>(0,0) << " " << p1.at<float>(0,1) << " " << p2.at<float>(0,0) << " " << p2.at<float>(0,1) << '\n';
    
    // std::cin.get();
    std::cout << p1(cv::Rect(0,0,2,5)) << '\n';
    std::cout << p2(cv::Rect(0,0,2,5)) << '\n';
    std::cin.get();

    const cv::Mat E = cv::findEssentialMat(p1, p2, f1.mK, cv::RANSAC, 0.99, 0.005, 100, mask);
    // const cv::Mat E = cv::findFundamentalMat(p1, p2, cv::FM_RANSAC, 1, 0.99, 100, mask);
    cur_row = 0;
    for(int i = 0; i < p1.rows; i++) {
        if(mask[i]){
            filteredKPmat.at<float>(cur_row,0) = p1.at<float>(i,0);
            filteredKPmat.at<float>(cur_row,1) = p1.at<float>(i,1);
            filteredKPmat.at<float>(cur_row,2) = p2.at<float>(i,0);
            filteredKPmat.at<float>(cur_row,3) = p2.at<float>(i,1);


            // p1.at<float>(cur_row,0) = p1n.at<float>(i,0);
            // p1.at<float>(cur_row,1) = p1n.at<float>(i,1);
            // p2.at<float>(cur_row,0) = p2n.at<float>(i,0);
            // p2.at<float>(cur_row,1) = p2n.at<float>(i,1);
            cur_row++;
        }
    }

    // p1.resize(cur_row);
    // p2.resize(cur_row);
    filteredKPmat.resize(cur_row);
    // std::cout << filteredKPmat.rows << '\n';

    // cv::hconcat(p1(cv::Rect(0,0,2,p1.rows)), p2(cv::Rect(0,0,2,p2.rows)), filteredKPmat);

    cv::Mat Rt = extractRt(E);
    // std::cout << mask.size() << " " << filteredKPmat.rows << '\n';
    
    // std::cout << Rt << '\n';
    // last_des = des;
    // last_kps = kps;
    return {filteredKPmat, Rt};
}
