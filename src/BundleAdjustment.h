#pragma once
#include <vector>
#include "SFMTypes.h"

struct BAResult {
    std::vector<CameraPose> cameras;
    std::vector<Eigen::Vector3f> points;
    double initialMeanReprojErrorPixels = 0.0;
    double finalMeanReprojErrorPixels = 0.0;
    int iterationsUsed = 0;
};

BAResult runBundleAdjustment(const std::vector<CameraPose>& initialCameras,
    const std::vector<Eigen::Vector3f>& initialPoints,
    const std::vector<Observation>& observations,
    float focalLengthPixels,
    int maxIterations = 20,
    float huberDeltaPixels = 4.0f);

struct FilterResult {
    std::vector<Observation> filteredObservations;
    std::vector<bool> pointKept;      // 长度=points.size()，标记每个点是否保留(观测数>=2)
    int numObservationsRemoved = 0;
    int numPointsRemoved = 0;
};

// BA收敛后，识别并剔除残差仍然过大的离群观测(以及因此丢失足够约束的点)。
// Huber核函数只是防止离群点在优化过程中"绑架"整体解，并不会真的移除它们——
// 这一步才是把它们从数据集里清理掉，配合再跑一轮BA，能拿到真正干净、可信的最终误差数字。
FilterResult filterOutliers(const std::vector<CameraPose>& cameras,
    const std::vector<Eigen::Vector3f>& points,
    const std::vector<Observation>& observations,
    float focalLengthPixels,
    float outlierThresholdPixels = 10.0f);