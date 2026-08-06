#pragma once
#include "FeatureSet.h"
#include <vector>

struct Match
{
	int idx1;		// 图1特征点下标
	int idx2;		// 图2特征点下标
	float distance; // 两个描述子距离，已开方
};

// 暴力匹配 + lowe，参数里rationThresh即阈值
std::vector<Match> matchBruteForce(const FeatureSet& f1, const FeatureSet& f2, float ratioThresh = 0.75f);