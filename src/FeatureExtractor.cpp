// SIFT特征提取器

#include "FeatureExtractor.h"
#include <cstring>
#include <iostream>

FeatureSet extractSIFTFeatures(const cv::Mat& image)
{
	// SIFT 算法要求传入的是灰度图
	cv::Mat gray;
	if (image.channels() == 3)
		cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);	// 转换成灰度图并保存进gray
	else
		gray = image;

	cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptorsMat;	// 描述子矩阵
	sift->detectAndCompute(gray, cv::noArray(), keypoints, descriptorsMat);

	FeatureSet fs;
	fs.numFeatures = static_cast<int>(keypoints.size());

	if (fs.numFeatures == 0)
	{
		std::cerr << "[FeatureExtractor] Warnig: No feature points were extracted!" << std::endl;
		return fs;
	}

	// 断言检查
	CV_Assert(descriptorsMat.type() == CV_32F);	// descriptorsMat 类型必须是 CV_32F
	CV_Assert(descriptorsMat.cols == FeatureSet::DESC_DIM);
	CV_Assert(descriptorsMat.isContinuous());

	fs.kpX.resize(fs.numFeatures);
	fs.kpY.resize(fs.numFeatures);
	for (int i = 0; i < fs.numFeatures; i++)
	{
		fs.kpX[i] = keypoints[i].pt.x;
		fs.kpY[i] = keypoints[i].pt.y;
	}

	// 将 cv::Mat 的连续内存拍平为扁平数组
	fs.descriptors.resize(static_cast<rsize_t>(fs.numFeatures) * FeatureSet::DESC_DIM);	// rsize_t = size_t
	std::memcpy(fs.descriptors.data(), descriptorsMat.ptr<float>(0), fs.descriptors.size() * sizeof(float));	// copy 建立在连续内存之上

	// 打印前 5 个描述子数值，看看是否正常
	std::cout << "[Debug] First 5 descriptor values in fs: ";
	for (int i = 0; i < 5 && i < fs.descriptors.size(); ++i) 
	{
		std::cout << fs.descriptors[i] << " ";
	}
	std::cout << std::endl;

	fs.cvKeypoints = std::move(keypoints);
	std::cout << "[FeatureExtractor] Extracted " << fs.numFeatures << " feature points." << std::endl;

	return fs;
}