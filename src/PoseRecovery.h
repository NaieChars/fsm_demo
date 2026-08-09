#pragma once
#include <vector>
#include <Eigen/Dense>

// 从本质矩阵E分解出相机2相对相机1的旋转R和平移方向t
// pts1, pts2一批归一化坐标对，通常传入RANSAC筛出的内点
// 返回true表示成功找到一组合理的(R,t)；如果四种组合都不合理返回false
bool recoverPose(const Eigen::Matrix3f& E,
    const std::vector<Eigen::Vector2f>& pts1,
    const std::vector<Eigen::Vector2f>& pts2,
    Eigen::Matrix3f& outR,
    Eigen::Vector3f& outT);

