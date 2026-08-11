#pragma once
#include <vector>
#include <Eigen/Dense>
#include "CameraIntrinsics.h"

struct PnPResult 
{
    Eigen::Matrix3f R;
    Eigen::Vector3f t;
    std::vector<int> inlierIndices; // 在输入的objectPoints/imagePoints里的下标(PnP内部RANSAC筛出的内点)
    bool success = false;
};

// 用PnP(Perspective-n-Point)求解新视角相机相对世界坐标系的位姿
// objectPoints: 已知的3D点(世界坐标系，也就是相机1坐标系下的坐标)
// imagePoints:  这些3D点在新视角图像里对应的像素坐标(未去畸变的原始像素坐标，PnP内部用的K和畸变系数处理)
// intr:         新视角相机的内参
PnPResult solvePnPForNewView(const std::vector<Eigen::Vector3f>& objectPoints,
    const std::vector<Eigen::Vector2f>& imagePoints,
    const CameraIntrinsics& intr);