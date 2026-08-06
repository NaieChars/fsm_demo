// 匹配阶段
// 暂且为暴力匹配加lowe的比率检测

#include "FeatureMatcher.h"
#include <limits>
#include <cmath>

// 计算两个128维描述子之间的欧氏距离的平方
static inline float squaredL2Distance(const float* a, const float* b, int dim)
{
	float sum = 0.0f;
	for (int i = 0; i < dim; i++)
	{
		float diff = a[i] - b[i];
		sum += diff * diff;
	}
	return sum;
}

// 朴素双层 for 循环与 cuda kernel 对应，方便迁移
std::vector<Match> matchBruteForce(const FeatureSet& f1, const FeatureSet& f2, float ratioThresh)
{
	std::vector<Match> matches;
	const int dim = FeatureSet::DESC_DIM;

	for (int i = 0; i < f1.numFeatures; i++)
	{
		const float* descA = &f1.descriptors[static_cast<size_t>(i) * dim];

		float bestDist = std::numeric_limits<float>::max();		// 最近邻距离平方
		float secondDist = std::numeric_limits<float>::max();	// 次近邻距离平方
		int bestIdx = -1;

		for (int j = 0; j < f2.numFeatures; j++)
		{
			const float* descB = &f2.descriptors[static_cast<size_t>(j) * dim];
			float d = squaredL2Distance(descA, descB, dim);

			if (d < bestDist)
			{
				secondDist = bestDist;
				bestDist = d;
				bestIdx = j;
			}
			else if (d < secondDist)
				secondDist = d;
		}

		// ratio test
		if (bestIdx >= 0 && bestDist < ratioThresh * ratioThresh * secondDist)
			matches.push_back({ i, bestIdx, std::sqrt(bestDist) });
	}

	return matches;
}