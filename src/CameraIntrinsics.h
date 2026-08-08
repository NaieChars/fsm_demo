/**
 * CameraIntrinsics.h
 * 相机内参结构体，包含 fx/fy/cx/cy、3x3 内参矩阵 K、逆矩阵 Kinv，
 * 以及 OpenCV 风格的 5 畸变系数 (k1,k2,p1,p2,k3)。
 * 提供：FOV 近似内参估算、像素归一化坐标转换、YAML 文件读写。
 */
#pragma once
#include <Eigen/Dense>
#include <array>
#include <string>

struct CameraIntrinsics
{
	float fx, fy, cx, cy;
	Eigen::Matrix3f K;
	Eigen::Matrix3f Kinv;

	// 畸变系数: k1, k2, p1, p2, k3 (OpenCV标准顺序)
	// k1,k2,k3 是径向畸变系数; p1,p2 是切向畸变系数
	std::array<float, 5> distCoeffs = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

	// 标定时使用的图像分辨率，用于后续加载时核对当前图像分辨率是否与标定时一致
	int calibWidth = 0;
	int calibHeight = 0;

	bool hasDistortion() const
	{
		for (float c : distCoeffs)
			if (c != 0.0f) return true;
		return false;
	}
};

// 用图像分辨率 + 近似水平视场角，估算一个粗略的相机内参矩阵K（无畸变系数）
// 在标定失败时作为备用方案
CameraIntrinsics approximateIntrinsics(int imageWidth, int imageHeight, float fovDegrees = 65.0f);

// 把像素坐标(px,py)转换成归一化相机坐标(不考虑畸变，纯针孔模型)
Eigen::Vector2f pixelToNormalized(const CameraIntrinsics& intr, float px, float py);

// 将标定结果保存到文本文件
bool saveIntrinsicsToFile(const CameraIntrinsics& intr, const std::string& path);

// 从本地读取标定结果
bool loadIntrinsicsFromFile(CameraIntrinsics& intr, const std::string& path);