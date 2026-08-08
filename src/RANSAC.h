#pragma once
#include <vector>
#include <Eigen/Dense>

struct RansacResult
{
	Eigen::Matrix3f E;				// 最终估计的 E
	std::vector<int> inlierIndices;	// 内点在原始pts1/pts2数组里的下标
	int iterationsUsed = 0;			// 实际进行的迭代次数
};

// 用RANSAC鲁棒估计本质矩阵，剔除错误匹配(outlier)
// pts1, pts2:         去畸变归一化坐标对
// thresholdPixels:    判定内点的误差阈值
// focalLengthPixels:  用于把像素阈值换算成归一化坐标系下的阈值(两者通过焦距关联)
// confidence:         RANSAC希望达到的置信度(至少采样到一组全内点8点组合的概率)，常用0.99
// maxIterations:      迭代次数上限(防止inlier比例过低时无限循环)
// randomSeed:         随机数种子，这里先固定值便于调试时结果可复现；正式使用可传time(nullptr)之类随机种子
RansacResult ransacEssentialMatrix(const std::vector<Eigen::Vector2f>& pts1,
    const std::vector<Eigen::Vector2f>& pts2,
    float thresholdPixels,
    float focalLengthPixels,
    float confidence = 0.99f,
    int maxIterations = 2000,
    unsigned int randomSeed = 42);