#include "PoseRecovery.h"
#include "Triangulation.h"
#include <iostream>
#include <algorithm>

// cheirality check
static int countPositiveDepth(const Eigen::Matrix3f& R, const Eigen::Vector3f& t,
    const std::vector<Eigen::Vector2f>& pts1,
    const std::vector<Eigen::Vector2f>& pts2,
    int maxCheck)
{
    ProjectionMatrix P1;
    P1 << Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero();

    ProjectionMatrix P2;
    P2.block<3, 3>(0, 0) = R;
    P2.block<3, 1>(0, 3) = t;

    int count = 0;
    int n = std::min(static_cast<int>(pts1.size()), maxCheck);
    for (int i = 0; i < n; ++i) 
    {
        Eigen::Vector3f X = triangulatePointDLT(P1, P2, pts1[i], pts2[i]);

        float depth1 = X.z();
        float depth2 = (R * X + t).z();

        if (depth1 > 0.0f && depth2 > 0.0f) 
        {
            count++;
        }
    }
    return count;
}

bool recoverPose(const Eigen::Matrix3f& E,
    const std::vector<Eigen::Vector2f>& pts1,
    const std::vector<Eigen::Vector2f>& pts2,
    Eigen::Matrix3f& outR,
    Eigen::Vector3f& outT)
{
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    // SVD 分解后，获取 UV
    Eigen::Matrix3f U = svd.matrixU();
    Eigen::Matrix3f V = svd.matrixV();

    // 修正矩阵方向，保证 det > 0 ,修正第三列等价于 U = -U
    if (U.determinant() < 0.0f) 
    {
        U.col(2) *= -1.0f;
    }
    if (V.determinant() < 0.0f) 
    {
        V.col(2) *= -1.0f;
    }

    Eigen::Matrix3f W;
    W << 0, -1, 0,
        1, 0, 0,
        0, 0, 1;

    // 计算 R_1 = UWV^T 和 R_2 = UW^TV^T 两个可能旋转
    Eigen::Matrix3f R1 = U * W * V.transpose();
    Eigen::Matrix3f R2 = U * W.transpose() * V.transpose();
    Eigen::Vector3f t = U.col(2);

    // 生成4种候选
    struct Candidate { Eigen::Matrix3f R; Eigen::Vector3f t; };
    std::vector<Candidate> candidates = 
    {
        { R1,  t }, { R1, -t }, { R2,  t }, { R2, -t }
    };

    const int maxCheck = std::min(static_cast<int>(pts1.size()), 50);
    int bestCount = -1;
    int bestIdx = -1;

    for (int i = 0; i < 4; i++)
    {
        int cnt = countPositiveDepth(candidates[i].R, candidates[i].t, pts1, pts2, maxCheck);
        std::cout << "[PoseRecovery] Candidate" << i << " positive depth points: " << cnt << " / " << maxCheck << std::endl;
        if (cnt > bestCount) 
        {
            bestCount = cnt;
            bestIdx = i;
        }
    }

    if (bestIdx < 0) 
    {
        std::cerr << "[PoseRecovery] Error: No valid pose candidate found." << std::endl;
        return false;
    }
    if (bestCount < maxCheck / 2) 
    {
        std::cerr << "[PoseRecovery] Warning: Less than half of the points have positive depth. The recovered pose may be unreliable." << std::endl;
    }

    outR = candidates[bestIdx].R;
    outT = candidates[bestIdx].t;
    return true;
}