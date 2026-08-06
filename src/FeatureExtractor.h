// SIFT特征提取器

#pragma once
#include <opencv2/opencv.hpp>
#include "FeatureSet.h"

// 对输入图像做SIFT特征提取，返回打包好的FeatureSet
FeatureSet extractSIFTFeatures(const cv::Mat& image);