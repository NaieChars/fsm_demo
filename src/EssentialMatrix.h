/**
 * EssentialMatrix.h
 * 
 * 归一化八点法求本质矩阵
 */

#pragma once
#include <vector>
#include <Eigen/Dense>

// 归一化八点法求本质矩阵
// pts1, pts2: 去畸变后的归一化相机坐标对，一一对应，size必须 >= 8且相等
Eigen::Matrix3f estimateEssentialMatrixLinear(const std::vector<Eigen::Vector2f>& pts1,
                                              const std::vector<Eigen::Vector2f>& pts2);
