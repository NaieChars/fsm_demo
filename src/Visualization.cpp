#include "Visualization.h"
#include <iostream>

void visualizeMatches(const cv::Mat& img1, const FeatureSet& f1,
                      const cv::Mat& img2, const FeatureSet& f2,
                      const std::vector<Match>& matches,
                      const std::string& windowName)
{
    std::vector<cv::DMatch> cvMatches;  // cv::DMatch 匹配点对
    cvMatches.reserve(matches.size());
    for (const auto& m : matches)
        cvMatches.emplace_back(m.idx1, m.idx2, m.distance);

    cv::Mat outImg;
    cv::drawMatches(img1, f1.cvKeypoints, img2, f2.cvKeypoints, cvMatches, outImg,
                    cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::resizeWindow(windowName, 1280, 720);

    cv::imshow(windowName, outImg);
    std::cout << "[Visualization] Displaying " << matches.size() << " filtered pairs" << std::endl;
    cv::waitKey(0);
}

bool checkCalibrationResolution(const cv::Mat& img1, const cv::Mat& img2, const CameraIntrinsics& intr)
{
    if (intr.calibWidth <= 0 || intr.calibHeight <= 0) {
        // 内参未记录标定分辨率，无法检查
        return true; // 认为通过（不做检查）
    }

    bool img1Match = (img1.cols == intr.calibWidth && img1.rows == intr.calibHeight);
    bool img2Match = (img2.cols == intr.calibWidth && img2.rows == intr.calibHeight);

    if (!img1Match || !img2Match) {
        std::cerr << "\n[Warning] Resolution mismatch! Calibration resolution was "
            << intr.calibWidth << "x" << intr.calibHeight
            << ", but current images are img1:" << img1.cols << "x" << img1.rows
            << " img2:" << img2.cols << "x" << img2.rows << std::endl;
        std::cerr << "[Warning] Continuing may produce inaccurate results. "
            << "Please check if camera settings match the calibration setup.\n" << std::endl;
        return false;
    }
    else {
        std::cout << "[main] Resolution check passed, matches calibration ("
            << intr.calibWidth << "x" << intr.calibHeight << ")" << std::endl;
        return true;
    }
}