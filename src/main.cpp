#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp> 
#include <Eigen/Dense>
#include <iostream>
#include <Windows.h>
#include <filesystem>

#include "FeatureSet.h"
#include "FeatureExtractor.h"
#include "FeatureMatcher.h"
#include "Visualization.h"
#include "CameraIntrinsics.h"
#include "Correspondences.h"
#include "RANSAC.h"

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

static bool robustLoadIntrinsics(CameraIntrinsics& intr, const std::string& filename)
{
    std::vector<std::string> candidates = {
        filename,
        "../" + filename,
        "../../" + filename,
        "../../../" + filename
    };
    for (const auto& path : candidates) {
        if (loadIntrinsicsFromFile(intr, path)) {
            return true;
        }
    }
    return false;
}


int main()
{
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR); 
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

    // 真实标定结果
    CameraIntrinsics intr1;
    if (!robustLoadIntrinsics(intr1, "camera_intrinsics.txt")) 
    {
        std::cerr << "Failed load camera_intrinsics.txt，Please confirm that the startup project is 'calibrate_tool' and build the executable." << std::endl;
        return -1;
    }
    CameraIntrinsics intr2 = intr1;
    // 分辨率一致性检测
    checkCalibrationResolution(img1, img2, intr1);

    // --------- 特征提取 ---------
    FeatureSet f1 = extractSIFTFeatures(img1);
    FeatureSet f2 = extractSIFTFeatures(img2);

    // --------- 特征匹配 + ratio test ---------
    std::vector<Match> matches = matchBruteForce(f1, f2, 0.75f);
    std::cout << "vertified " << matches.size() << " pairs keypoints" << std::endl;

    // ----- 构建去畸变后的归一化坐标
    NormalizedCorrespondences corr = buildNormalizedCorrespondences(f1, f2, matches, intr1, intr2);
    std::cout << "\n[Correspondences] First 5 pairs of undistorted normalized coordinates:" << std::endl;
    for (int i = 0; i < (std::min)(5, static_cast<int>(corr.pts1.size())); ++i) 
    {
        std::cout << "  pair " << i << ": p1=(" << corr.pts1[i].x() << ", " << corr.pts1[i].y()
            << ")  p2=(" << corr.pts2[i].x() << ", " << corr.pts2[i].y() << ")" << std::endl;
    }

    // ---- RANSAC鲁棒估计本质矩阵，剔除错误匹配 -------------
    RansacResult ransacResult = ransacEssentialMatrix(corr.pts1, corr.pts2, 1.5f, intr1.fx);

    std::cout << "\n[Main] Estimated essential matrix E:\n" << ransacResult.E << std::endl;

    // 把RANSAC筛出的内点下标，映射回原始的Match列表，方便复用现有的可视化函数
    std::vector<Match> inlierMatches;
    inlierMatches.reserve(ransacResult.inlierIndices.size());
    for (int idx : ransacResult.inlierIndices) 
    {
        inlierMatches.push_back(matches[idx]);
    }

    // ---------- 可视化 ----------
    visualizeMatches(img1, f1, img2, f2, matches, "All Matches (before RANSAC)");
    visualizeMatches(img1, f1, img2, f2, inlierMatches, "RANSAC Inliers");

    return 0;
}
