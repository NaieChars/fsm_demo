/**
 * Correspondences.h
 * 把像素匹配点转换成归一化相机坐标（Z=1平面），供对极几何/本质矩阵求解使用
 * 依赖 CameraIntrinsics 做内参剥离，纯针孔模型（未含畸变校正）。
 */

#pragma once
#include <vector>
#include <Eigen/Dense>
#include "FeatureSet.h"
#include "FeatureMatcher.h"
#include "CameraIntrinsics.h"

struct NormalizedCorrespondences 
{
    std::vector<Eigen::Vector2f> pts1; // 图1中，每个匹配点的归一化相机坐标
    std::vector<Eigen::Vector2f> pts2; // 图2中，对应点的归一化相机坐标
};

// 根据匹配结果和两张图各自的内参，构建归一化坐标对
NormalizedCorrespondences buildNormalizedCorrespondences(
    const FeatureSet& f1, const FeatureSet& f2,
    const std::vector<Match>& matches,
    const CameraIntrinsics& intr1, const CameraIntrinsics& intr2);