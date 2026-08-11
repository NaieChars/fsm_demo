// 增量式 SFM
// 引入第三个相机，借助相机1的3D点和新相机投影下的二维点
#include "PnPSolver.h"
#include <opencv2/opencv.hpp>
#include <iostream>

PnPResult solvePnPForNewView(const std::vector<Eigen::Vector3f>& objectPoints,
    const std::vector<Eigen::Vector2f>& imagePoints,
    const CameraIntrinsics& intr)
{
    PnPResult result;

    const int n = static_cast<int>(objectPoints.size());
    if (n < 6) 
    {
        std::cerr << "[PnPSolver] Too few valid 3D-2D correspondences(" << n << "points, at least 6 recommended), PnP result may be unreliable." << std::endl;
        result.success = false;
        return result;
    }

    // 转换成 OpenCV 格式，不用 Eigen
    std::vector<cv::Point3f> cvObjectPoints(n);
    std::vector<cv::Point2f> cvImagePoints(n);
    for (int i = 0; i < n; ++i) 
    {
        cvObjectPoints[i] = cv::Point3f(objectPoints[i].x(), objectPoints[i].y(), objectPoints[i].z());
        cvImagePoints[i] = cv::Point2f(imagePoints[i].x(), imagePoints[i].y());
    }

    cv::Mat K = (cv::Mat_<double>(3, 3) <<
        intr.fx, 0.0, intr.cx,
        0.0, intr.fy, intr.cy,
        0.0, 0.0, 1.0);

    cv::Mat D = (cv::Mat_<double>(1, 5) <<
        intr.distCoeffs[0], intr.distCoeffs[1],
        intr.distCoeffs[2], intr.distCoeffs[3],
        intr.distCoeffs[4]);

    cv::Mat rvec, tvec;
    std::vector<int> inliers;

    bool ok = cv::solvePnPRansac(
        cvObjectPoints, cvImagePoints, K, D,
        rvec, tvec,
        false,
        200,
        3.0f,
        0.99,
        inliers,
        cv::SOLVEPNP_ITERATIVE);

    if (!ok || inliers.empty()) 
    {
        std::cerr << "[PnPSolver] solvePnPRansac failed" << std::endl;
        result.success = false;
        return result;
    }

    cv::Mat Rmat;
    cv::Rodrigues(rvec, Rmat);

    // OpenCV 转回 Eigen
    Eigen::Matrix3f R;
    for (int r = 0; r < 3; ++r) 
    {
        for (int c = 0; c < 3; ++c) 
        {
            R(r, c) = static_cast<float>(Rmat.at<double>(r, c));
        }
    }

    Eigen::Vector3f t(
        static_cast<float>(tvec.at<double>(0)),
        static_cast<float>(tvec.at<double>(1)),
        static_cast<float>(tvec.at<double>(2)));

    result.R = R;
    result.t = t;
    result.inlierIndices = inliers;
    result.success = true;

    std::cout << "[PnPSolver] PnP solved successfully. Inliers: " << inliers.size() << " / " << n << std::endl;

    return result;
}