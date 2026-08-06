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

    cv::namedWindow("SIFT Matches", cv::WINDOW_NORMAL); // 允许窗口自由缩放
    cv::resizeWindow("SIFT Matches", 1280, 720);    // 强制窗口大小

    cv::imshow(windowName, outImg);
    std::cout << "[Visualization] Displaying " << matches.size() << " filtered pairs" << std::endl;
    cv::waitKey(0);
}

