#define _USE_MATH_DEFINES	// 放开非标准扩展，让 M_PI 生效
#include "CameraIntrinsics.h"
#include <cmath>
#include <fstream>
#include <iostream>

CameraIntrinsics approximateIntrinsics(int imageWidth, int imageHeight, float fovDegrees)
{
	CameraIntrinsics intr;

	float fovRad = fovDegrees * static_cast<float>(M_PI) / 180.0f;
	// 计算焦距 fx, fy
	intr.fx = imageWidth / (2.0f * tan(fovRad / 2.0f));
	intr.fy = intr.fx;
	// 主点位置
	intr.cx = imageWidth / 2.0f;
	intr.cy = imageHeight / 2.0f;

	// 构建内参矩阵
	intr.K << intr.fx, 0.0f,    intr.cx,
			  0.0f,    intr.fy, intr.cy,
			  0.0f,    0.0f,    1.0f;
	intr.Kinv = intr.K.inverse();	// 逆矩阵 Kinv
	return intr;
}

Eigen::Vector2f pixelToNormalized(const CameraIntrinsics& intr, float px, float py)
{
	float xn = (px - intr.cx) / intr.fx;
	float yn = (py - intr.cy) / intr.fy;
	return Eigen::Vector2f(xn, yn);
}

bool saveIntrinsicsToFile(const CameraIntrinsics& intr, const std::string& path)
{
	std::ofstream ofs(path);
	if (!ofs.is_open())
	{
		std::cerr << "[CameraIntrinsics] Failed to write to file: " << path << std::endl;
		return false;
	}

	// 简单的纯文本格式，一行一个数值组，顺序固定：
	// 第1行: fx fy cx cy
	// 第2行: k1 k2 p1 p2 k3
	// 第3行: calibWidth calibHeight
	ofs << intr.fx << " " << intr.fy << " " << intr.cx << " " << intr.cy << "\n";
	ofs << intr.distCoeffs[0] << " " << intr.distCoeffs[1] << " "
		<< intr.distCoeffs[2] << " " << intr.distCoeffs[3] << " "
		<< intr.distCoeffs[4] << "\n";
	ofs << intr.calibWidth << " " << intr.calibHeight << "\n";

	ofs.close();
	std::cout << "[CameraIntrinsics] Calibration results saved to: " << path << std::endl;
	return true;
}

bool loadIntrinsicsFromFile(CameraIntrinsics& intr, const std::string& path)
{
	std::ifstream ifs(path);
	if (!ifs.is_open())
	{
		std::cerr << "[CameraIntrinsics] Can not read the file: " << path << std::endl;
		return false;
	}

	ifs >> intr.fx >> intr.fy >> intr.cx >> intr.cy;
	ifs >> intr.distCoeffs[0] >> intr.distCoeffs[1]
		>> intr.distCoeffs[2] >> intr.distCoeffs[3] >> intr.distCoeffs[4];

	if (ifs.fail()) 
	{
		std::cerr << "[CameraIntrinsics] File format error: " << path << std::endl;
		return false;
	}

	if (!(ifs >> intr.calibWidth >> intr.calibHeight)) 
	{
		intr.calibWidth = 0;
		intr.calibHeight = 0;
		ifs.clear();
	}

	intr.K << intr.fx, 0.0f,    intr.cx,
			  0.0f,    intr.fy, intr.cy,
			  0.0f,    0.0f,    1.0f;
	intr.Kinv = intr.K.inverse();

	std::cout << "[CameraIntrinsics] Calibration results loaded from file: " << path << std::endl;
	return true;
}