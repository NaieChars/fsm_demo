/**
 * CameraCalibration.h
 * 严格棋盘格标定：求解完整相机内参
 */
#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "CameraIntrinsics.h"

// 对一组棋盘格标定照片跑相机标定
// imagePaths:    标定照片的文件路径列表
// boardSize:     棋盘格内部角点数(横向, 纵向)
// squareSizeMM:  棋盘格每个格子的真实物理边长(mm)
// outIntr:       输出标定得到的相机内参(含畸变系数)
// return: 平均重投影误差(像素)。如果标定失败返回负数。
double runCameraCalibration(const std::vector<std::string>& imagePaths,
    cv::Size boardSize,
    float squareSizeMM,
    CameraIntrinsics& outIntr);