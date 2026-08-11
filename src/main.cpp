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
#include "PoseRecovery.h"
#include "Triangulation.h"
#include "PointCloudExport.h"
#include "PnPSolver.h"
#include "IncrementalSFM.h"

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
    std::vector<std::string> candidates = 
    {
        filename,
        "../" + filename,
        "../../" + filename,
        "../../../" + filename
    };
    for (const auto& path : candidates) 
    {
        if (loadIntrinsicsFromFile(intr, path)) 
            return true;
    }
    return false;
}


int main()
{
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR); 
    SetConsoleCP(CP_UTF8); 
    SetConsoleOutputCP(CP_UTF8);

    CameraIntrinsics intr;

    if (!robustLoadIntrinsics(intr, "camera_intrinsics_blender.txt")) {
        std::cerr << "Failed to load camera_intrinsics.txt. Please make sure calibrate_tool has been executed and the file has been generated." << std::endl;
        return -1;
    }

    int imagesProcessed = 0;
    std::vector<ColoredPoint3D> pointCloud = runIncrementalSFM(intr, imagesProcessed);

    std::cout << "\n[Main] Successfully registered " << imagesProcessed << " views. Final point cloud size: " << pointCloud.size() << " points." << std::endl;

    savePointCloudPLY(pointCloud, "sparse_pointcloud.ply");

    return 0;
}