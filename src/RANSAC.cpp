#include "RANSAC.h"
#include "EssentialMatrix.h"
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <limits>

// sampson 距离*
static float sampsonDistanceSquared(const Eigen::Matrix3f& E,
    const Eigen::Vector2f& p1,
    const Eigen::Vector2f& p2)
{
    Eigen::Vector3f x1(p1.x(), p1.y(), 1.0f);
    Eigen::Vector3f x2(p2.x(), p2.y(), 1.0f);

    // 极线方向
    Eigen::Vector3f Ex1 = E * x1;
    Eigen::Vector3f Etx2 = E.transpose() * x2;

    // 计算代数误差
    float numerator = x2.dot(Ex1);
    numerator = numerator * numerator;

    // 计算梯度模长
    float denominator = Ex1(0) * Ex1(0) + Ex1(1) * Ex1(1)   // a^2 + b^2
        + Etx2(0) * Etx2(0) + Etx2(1) * Etx2(1);            // a'^2 + b'^2

    // sampson 距离
    return numerator / (denominator + 1e-12f);
}

// 给定一个 E 假设，统计所有匹配点里多少是内点
static std::vector<int> countInliers(const Eigen::Matrix3f& E,
    const std::vector<Eigen::Vector2f>& pts1,
    const std::vector<Eigen::Vector2f>& pts2,
    float thresholdSq)
{
    std::vector<int> inliers;
    inliers.reserve(pts1.size());

    for (int i = 0; i < static_cast<int>(pts1.size()); i++)
    {
        float d = sampsonDistanceSquared(E, pts1[i], pts2[i]);
        if (d < thresholdSq)
            inliers.push_back(i);
    }
    return inliers;
}

RansacResult ransacEssentialMatrix(const std::vector<Eigen::Vector2f>& pts1,
    const std::vector<Eigen::Vector2f>& pts2,
    float thresholdPixels,
    float focalLengthPixels,
    float confidence,
    int maxIterations,
    unsigned int randomSeed)
{
    const int n = static_cast<int>(pts1.size());
    const int sampleSize = 8;

    // 将像素坐标下的阈值转换成归一化坐标下的阈值
    // 注意sampson用的是平方误差，这里阈值也要平方
    float thresholdNormalized = thresholdPixels / focalLengthPixels;
    float thresholdSq = thresholdNormalized * thresholdNormalized;

    std::mt19937 rng(randomSeed);

    // 预生成所有迭代要用的随机采样索引，每次迭代互不依赖
    std::vector<std::vector<int>> allSamples(maxIterations);
    std::vector<int> allIndices(n);
    std::iota(allIndices.begin(), allIndices.end(), 0);

    for (int iter = 0; iter < maxIterations; iter++)
    {
        std::vector<int> sample(sampleSize);
        // std::sample: 从allIndices里无放回地随机取8个不重复的下标
        std::sample(allIndices.begin(), allIndices.end(), sample.begin(), sampleSize, rng);
        allSamples[iter] = std::move(sample);
    }

    // 主循环
    RansacResult best;
    int bestInlierCount = 0;
    int actualIterations = maxIterations;

    for (int iter = 0; iter < maxIterations; iter++)
    {
        const std::vector<int>& sampleIdx = allSamples[iter];
        std::vector<Eigen::Vector2f> subPts1(sampleSize), subPts2(sampleSize);
        for (int k = 0; k < sampleSize; ++k) 
        {
            subPts1[k] = pts1[sampleIdx[k]];
            subPts2[k] = pts2[sampleIdx[k]];
        }

        Eigen::Matrix3f Ehypothesis = estimateEssentialMatrixLinear(subPts1, subPts2);
        std::vector<int> inliers = countInliers(Ehypothesis, pts1, pts2, thresholdSq);

        if (static_cast<int>(inliers.size()) > bestInlierCount)
        {
            bestInlierCount = static_cast<int>(inliers.size());
            best.E = Ehypothesis;
            best.inlierIndices = inliers;

            // 自适应提前停止
            float w = static_cast<float>(bestInlierCount) / static_cast<float>(n);
            if (w > 0.0f && w < 1.0f)
            {
                // 计算分母（单次采样失败的对数概率）
                double wd = static_cast<double>(w);
                double denom = std::log(1.0 - std::pow(wd, sampleSize));
                const double kMinDenomMagnitude = 1e-12;
                if (denom < -kMinDenomMagnitude) //denom应小于0
                {
                    double requiredIters = std::log(1.0 - static_cast<double>(confidence)) / denom;
                    //还需的迭代次数
                    if (requiredIters < static_cast<double>(actualIterations)) 
                    {
                        int requiredItersInt = static_cast<int>(std::ceil(requiredIters));
                        actualIterations = std::max(requiredItersInt, iter + 1);
                    }
                }
            }
        }

        if (iter + 1 >= actualIterations)
        {
            actualIterations = iter + 1;
            break;
        }
    }

    std::cout << "[RANSAC] Completed! total iterations: " << actualIterations
        << ", best inlier count: " << bestInlierCount << " / " << n
        << " (inlier ratio: " << (100.0f * bestInlierCount / n) << "%)" << std::endl;

    if (bestInlierCount < sampleSize)
    {
        std::cerr << "[RANSAC] Warning: Best inlier count too small (<8), essential matrix estimation is unreliable" << std::endl;
        best.iterationsUsed = actualIterations;
        return best;
    }

    // 用全部内点重新拟合一次E
    std::vector<Eigen::Vector2f> inlierPts1, inlierPts2;
    inlierPts1.reserve(best.inlierIndices.size());
    inlierPts2.reserve(best.inlierIndices.size());
    for (int idx : best.inlierIndices)
    {
        inlierPts1.push_back(pts1[idx]);
        inlierPts2.push_back(pts2[idx]);
    }
    best.E = estimateEssentialMatrixLinear(inlierPts1, inlierPts2);
    best.iterationsUsed = actualIterations;

    return best;
}