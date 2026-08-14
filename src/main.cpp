#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp> 
#include <Eigen/Dense>
#include <iostream>
#include <Windows.h>
#include <filesystem>
#include <algorithm>

#include "CameraIntrinsics.h"
#include "IncrementalSFM.h"
#include "BundleAdjustment.h"
#include "PointCloudExport.h"

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
    if (!robustLoadIntrinsics(intr, "camera_intrinsics.txt")) {
        std::cerr << "无法加载 camera_intrinsics.txt，请确认已运行 calibrate_tool 并生成该文件" << std::endl;
        return -1;
    }

    int imagesProcessed = 0;
    SFMResult sfmResult = runIncrementalSFM(intr, imagesProcessed);

    std::cout << "\n[Main] 成功注册 " << imagesProcessed << " 个视角，增量式重建点云总点数: "
        << sfmResult.pointCloud.size() << std::endl;

    savePointCloudPLY(sfmResult.pointCloud, "sparse_pointcloud_before_ba.ply");

    // ---- Bundle Adjustment 全局优化 ----
    std::vector<Eigen::Vector3f> points;
    points.reserve(sfmResult.pointCloud.size());
    for (const auto& cp : sfmResult.pointCloud) {
        points.push_back(cp.position);
    }

    BAResult baResult = runBundleAdjustment(sfmResult.cameras, points, sfmResult.observations, intr.fx);

    std::cout << "\n[Main] BA前平均重投影误差: " << baResult.initialMeanReprojErrorPixels << " 像素" << std::endl;
    std::cout << "[Main] BA后平均重投影误差: " << baResult.finalMeanReprojErrorPixels << " 像素" << std::endl;

    std::vector<ColoredPoint3D> optimizedPointCloud = sfmResult.pointCloud;
    for (size_t i = 0; i < optimizedPointCloud.size(); ++i) {
        optimizedPointCloud[i].position = baResult.points[i];
    }
    savePointCloudPLY(optimizedPointCloud, "sparse_pointcloud_after_ba.ply");

    // ---- 过滤残差仍然过大的离群观测，再跑一轮BA做最后精修 ----
    FilterResult filterResult = filterOutliers(baResult.cameras, baResult.points, sfmResult.observations, intr.fx, 10.0f);
    std::cout << "\n[Main] 过滤掉 " << filterResult.numObservationsRemoved << " 条离群观测, "
        << filterResult.numPointsRemoved << " 个点因观测不足被剔除" << std::endl;

    BAResult finalBaResult = runBundleAdjustment(baResult.cameras, baResult.points, filterResult.filteredObservations, intr.fx);
    std::cout << "[Main] 精修后平均重投影误差: " << finalBaResult.finalMeanReprojErrorPixels << " 像素" << std::endl;

    std::vector<ColoredPoint3D> finalPointCloud;
    finalPointCloud.reserve(optimizedPointCloud.size());
    for (size_t i = 0; i < sfmResult.pointCloud.size(); ++i) {
        if (filterResult.pointKept[i]) {
            ColoredPoint3D cp = sfmResult.pointCloud[i];
            cp.position = finalBaResult.points[i];
            finalPointCloud.push_back(cp);
        }
    }
    savePointCloudPLY(finalPointCloud, "sparse_pointcloud_final.ply");

    return 0;
}