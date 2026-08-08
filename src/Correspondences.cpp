#include "Correspondences.h"
#include <opencv2/opencv.hpp>

// 把CameraIntrinsics转换成cv::undistortPoints需要的K矩阵和畸变系数向量(OpenCV格式，double类型)
static void toOpenCvKD(const CameraIntrinsics& intr, cv::Mat& K, cv::Mat& D)
{
    K = (cv::Mat_<double>(3, 3) <<
        intr.fx, 0.0, intr.cx,
        0.0, intr.fy, intr.cy,
        0.0, 0.0, 1.0);

    D = (cv::Mat_<double>(1, 5) <<
        intr.distCoeffs[0], intr.distCoeffs[1],
        intr.distCoeffs[2], intr.distCoeffs[3],
        intr.distCoeffs[4]);
}

NormalizedCorrespondences buildNormalizedCorrespondences(
    const FeatureSet& f1, const FeatureSet& f2,
    const std::vector<Match>& matches,
    const CameraIntrinsics& intr1, const CameraIntrinsics& intr2)
{
    NormalizedCorrespondences corr;
    const int n = static_cast<int>(matches.size());
    if (n == 0) return corr;

    // 组装两组原始像素坐标（未去畸变），供cv::undistortPoints批量处理
    std::vector<cv::Point2f> rawPts1(n), rawPts2(n);
    for (int i = 0; i < n; ++i) {
        rawPts1[i] = cv::Point2f(f1.kpX[matches[i].idx1], f1.kpY[matches[i].idx1]);
        rawPts2[i] = cv::Point2f(f2.kpX[matches[i].idx2], f2.kpY[matches[i].idx2]);
    }

    cv::Mat K1, D1, K2, D2;
    toOpenCvKD(intr1, K1, D1);
    toOpenCvKD(intr2, K2, D2);

    std::vector<cv::Point2f> undistPts1, undistPts2;

    // cv::undistortPoints 做的事：输入畸变后的像素坐标，输出去畸变+归一化后的理想相机坐标。
    cv::undistortPoints(rawPts1, undistPts1, K1, D1);
    cv::undistortPoints(rawPts2, undistPts2, K2, D2);

    corr.pts1.reserve(n);
    corr.pts2.reserve(n);
    for (int i = 0; i < n; ++i) {
        corr.pts1.emplace_back(undistPts1[i].x, undistPts1[i].y);
        corr.pts2.emplace_back(undistPts2[i].x, undistPts2[i].y);
    }

    return corr;
}