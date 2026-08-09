#pragma once
#include <Eigen/Dense>

// 3x4投影矩阵类型别名
using ProjectionMatrix = Eigen::Matrix<float, 3, 4>;

// 线性三角化(DLT法)：给定两个相机的投影矩阵和一对归一化图像坐标，求解对应的3D点
// 返回的3D点坐标系是相机1的坐标系(因为P1通常是[I|0])
Eigen::Vector3f triangulatePointDLT(const ProjectionMatrix& P1, const ProjectionMatrix& P2,
    const Eigen::Vector2f& x1, const Eigen::Vector2f& x2);