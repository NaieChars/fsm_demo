#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include "FeatureSet.h"
#include "FeatureMatcher.h"
#include "CameraIntrinsics.h"

void visualizeMatches(const cv::Mat& img1, const FeatureSet& f1, const cv::Mat& img2, const FeatureSet& f2,
					  const std::vector<Match>& matches, const std::string& windowName);

bool checkCalibrationResolution(const cv::Mat& img1, const cv::Mat& img2, const CameraIntrinsics& intr);