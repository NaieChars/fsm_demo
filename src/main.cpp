#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <Windows.h>
#include <filesystem>

#include "FeatureSet.h"
#include "FeatureExtractor.h"
#include "FeatureMatcher.h"
#include "Visualization.h"

// 简单地多路径读图，避免 VS 的 CMake 的工程目录不确定导致读图失败
static cv::Mat robustImread(const std::string& filename)
{
    std::vector<std::string> candidates =
    {
        filename,
        "../" + filename,
        "../../" + filename,
        "../../../" + filename
    };
    for (const auto& path : candidates) 
    {
        cv::Mat img = cv::imread(path);
        if (!img.empty()) 
        {
            std::cout << "[main] Read successfully: " << path << std::endl;
            return img;
        }
    }
    return cv::Mat(); // 全部失败，返回空Mat
}


int main()
{
    SetConsoleCP(CP_UTF8); 
    SetConsoleOutputCP(CP_UTF8);
    
    cv::Mat img1 = robustImread("data/img1.jpg");
    cv::Mat img2 = robustImread("data/img2.jpg");

    if (img1.empty() || img2.empty()) 
    {
        std::cerr << "Failed to read images. Please verify that data/img1.jpg and data/img2.jpg exist in the project root directory." << std::endl;
        return -1;
    }
    std::cout << "img1: " << img1.cols << "x" << img1.rows << std::endl;
    std::cout << "img2: " << img2.cols << "x" << img2.rows << std::endl;

    // --------- 特征提取 ---------
    FeatureSet f1 = extractSIFTFeatures(img1);
    FeatureSet f2 = extractSIFTFeatures(img2);

    // --------- 特征匹配 + ratio test ---------
    std::vector<Match> matches = matchBruteForce(f1, f2, 0.75f);
    std::cout << "vertified " << matches.size() << " pairs keypoints" << std::endl;

    // ---------- 可视化 ----------
    visualizeMatches(img1, f1, img2, f2, matches, "SIFT Matches");

    return 0;
}
