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

bool recoverPoseFromRansac(const RansacResult& ransacResult,
    const NormalizedCorrespondences& corr,
    Eigen::Matrix3f& R,
    Eigen::Vector3f& t,
    std::vector<Eigen::Vector2f>& inlierPts1,
    std::vector<Eigen::Vector2f>& inlierPts2);

std::vector<ColoredPoint3D> triangulateAndColorPointCloud(
    const std::vector<Eigen::Vector2f>& inlierPts1,
    const std::vector<Eigen::Vector2f>& inlierPts2,
    const Eigen::Matrix3f& R,
    const Eigen::Vector3f& t,
    const RansacResult& ransacResult,
    const std::vector<Match>& matches,
    const FeatureSet& f1,
    const cv::Mat& img1);


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

    // ---------- 真实标定结果 -------------
    CameraIntrinsics intr1;
    if (!robustLoadIntrinsics(intr1, "camera_intrinsics_blender.txt")) 
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

    // ----------E 分解两个相机的位姿(R,t) -----------
    Eigen::Matrix3f R;
    Eigen::Vector3f t;
    std::vector<Eigen::Vector2f> inlierPts1, inlierPts2;
    if (!recoverPoseFromRansac(ransacResult, corr, R, t, inlierPts1, inlierPts2))
        return -1;
    
    // ----------- 三角化并生成点云 ------------
    std::vector<ColoredPoint3D> pointCloud = triangulateAndColorPointCloud(
        inlierPts1, inlierPts2, R, t, ransacResult, matches, f1, img1);

    savePointCloudPLY(pointCloud, "sparse_pointcloud.ply");

    // ---------- 可视化 ----------
    visualizeMatches(img1, f1, img2, f2, matches, "All Matches (before RANSAC)");
    visualizeMatches(img1, f1, img2, f2, inlierMatches, "RANSAC Inliers");

    return 0;
}

bool recoverPoseFromRansac(const RansacResult& ransacResult,
    const NormalizedCorrespondences& corr,
    Eigen::Matrix3f& R,
    Eigen::Vector3f& t,
    std::vector<Eigen::Vector2f>& inlierPts1,
    std::vector<Eigen::Vector2f>& inlierPts2)
{
    // 提取内点坐标
    inlierPts1.clear();
    inlierPts2.clear();
    inlierPts1.reserve(ransacResult.inlierIndices.size());
    inlierPts2.reserve(ransacResult.inlierIndices.size());
    for (int idx : ransacResult.inlierIndices) {
        inlierPts1.push_back(corr.pts1[idx]);
        inlierPts2.push_back(corr.pts2[idx]);
    }

    // 从本质矩阵分解位姿
    if (!recoverPose(ransacResult.E, inlierPts1, inlierPts2, R, t)) {
        std::cerr << "[Main] 位姿恢复失败" << std::endl;
        return false;
    }

    std::cout << "\n[Main] 恢复出的相机2相对位姿:\nR =\n" << R << "\nt = " << t.transpose() << std::endl;
    return true;
}


std::vector<ColoredPoint3D> triangulateAndColorPointCloud(
    const std::vector<Eigen::Vector2f>& inlierPts1,
    const std::vector<Eigen::Vector2f>& inlierPts2,
    const Eigen::Matrix3f& R,
    const Eigen::Vector3f& t,
    const RansacResult& ransacResult,
    const std::vector<Match>& matches,
    const FeatureSet& f1,
    const cv::Mat& img1)
{
    // 构建投影矩阵
    ProjectionMatrix P1;
    P1 << Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero();
    ProjectionMatrix P2;
    P2.block<3, 3>(0, 0) = R;
    P2.block<3, 1>(0, 3) = t;

    std::vector<ColoredPoint3D> pointCloud;
    pointCloud.reserve(inlierPts1.size());

    for (size_t k = 0; k < inlierPts1.size(); ++k) {
        int idx = ransacResult.inlierIndices[k];         // 原始匹配索引
        Eigen::Vector3f X = triangulatePointDLT(P1, P2, inlierPts1[k], inlierPts2[k]);

        // Cheirality 检查：点必须在两台相机前方
        float depth1 = X.z();
        float depth2 = (R * X + t).z();
        if (depth1 <= 0.0f || depth2 <= 0.0f) continue;

        // 从图1中采样颜色（使用匹配点对应的关键点坐标）
        int px = static_cast<int>(f1.kpX[matches[idx].idx1]);
        int py = static_cast<int>(f1.kpY[matches[idx].idx1]);
        px = std::clamp(px, 0, img1.cols - 1);
        py = std::clamp(py, 0, img1.rows - 1);
        cv::Vec3b bgr = img1.at<cv::Vec3b>(py, px);

        ColoredPoint3D cp;
        cp.position = X;
        cp.r = bgr[2];   // BGR -> RGB
        cp.g = bgr[1];
        cp.b = bgr[0];
        pointCloud.push_back(cp);
    }

    std::cout << "[Main] Triangulation complete: " << pointCloud.size() << " / " << inlierPts1.size()
        << " points passed cheirality check" << std::endl;

    return pointCloud;
}