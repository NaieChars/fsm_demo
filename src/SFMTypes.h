// src/SFMTypes.h
#pragma once
#include <vector>
#include <Eigen/Dense>
#include "PointCloudExport.h"

struct CameraPose 
{
    Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
    Eigen::Vector3f t = Eigen::Vector3f::Zero();
};

// 一次观测：相机cameraIdx看到了点pointIdx，在该相机图像里的去畸变归一化坐标是uv
struct Observation 
{
    int cameraIdx;
    int pointIdx;
    Eigen::Vector2f uv;
};

struct SFMResult 
{
    std::vector<CameraPose> cameras;        // 每个视角的位姿(相机0固定为单位位姿)
    std::vector<ColoredPoint3D> pointCloud; // 重建出的点云
    std::vector<Observation> observations;  // 所有观测记录，供BundleAdjustment使用
};