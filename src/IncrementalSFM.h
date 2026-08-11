#pragma once
#include <vector>
#include "CameraIntrinsics.h"
#include "PointCloudExport.h"

// 增量式SFM主流程：依次尝试读取 data/img1.jpg, data/img2.jpg, ... 直到找不到下一张为止。
// intr:            相机内参(含畸变系数)，假设所有图片来自同一相机/同一份标定结果
// imagesProcessed: [输出] 实际成功注册位姿的图片数量(包含初始两张)
std::vector<ColoredPoint3D> runIncrementalSFM(const CameraIntrinsics& intr, int& imagesProcessed);