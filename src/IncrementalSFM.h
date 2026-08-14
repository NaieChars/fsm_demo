#pragma once
#include <vector>
#include "CameraIntrinsics.h"
#include "SFMTypes.h"

// 增量式SFM主流程：依次尝试读取 data/img1.jpg, data/img2.jpg, ... 直到找不到下一张为止。
//  前两张图做初始两视图重建；之后每张新图与"最近一次成功注册"的图做链式匹配，
// 通过PnP估计新视角位姿，并三角化出全新的3D点，逐步扩展点云。
// 每当一个已有3D点被后续新图重新观测到(用于PnP)时，也会记录这次新的观测，
// 让点的轨迹覆盖多个视角，供后续Bundle Adjustment使用。
// intr:            相机内参(含畸变系数)，假设所有图片来自同一相机/同一份标定结果
// imagesProcessed: [输出] 实际成功注册位姿的图片数量(包含初始两张)
SFMResult runIncrementalSFM(const CameraIntrinsics& intr, int& imagesProcessed);