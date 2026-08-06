// 特征点基础设置

#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

struct FeatureSet
{
	static constexpr int DESC_DIM = 128;

	int numFeatures = 0;

	// SoA布局，方便 CUDA 重构
	std::vector<float> kpX;	// 关键点的x坐标
	std::vector<float> kpY;
	std::vector<float> descriptors;	// 展平后的描述子，大小：numFeatures * DESC_DIM

	// 可视化需要，不参与计算
	std::vector<cv::KeyPoint> cvKeypoints;
};