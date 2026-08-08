#include "EssentialMatrix.h"
#include <cassert>

Eigen::Matrix3f estimateEssentialMatrixLinear(const std::vector<Eigen::Vector2f>& pts1,
    const std::vector<Eigen::Vector2f>& pts2)
{
    const int n = static_cast<int>(pts1.size());
    assert(pts1.size() == pts2.size());
    assert(n >= 8);

    // 构造 A 矩阵 (n x 9)，每一行对应一对匹配点的约束方程 
    Eigen::MatrixXf A(n, 9);
    for (int i = 0; i < n; i++)
    {
        float x = pts1[i].x(), y = pts1[i].y();
        float xp = pts2[i].x(), yp = pts2[i].y();

        A.row(i) << xp * x, xp* y, xp,
                    yp* x, yp* y, yp,
                    x, y, 1.0f;
    }

    // 构造E
    // SVD分解A,取最小奇异值对应的右奇异向量作为解
    Eigen::JacobiSVD<Eigen::MatrixXf> svdA(A, Eigen::ComputeFullV);
    Eigen::VectorXf e = svdA.matrixV().col(8);

    Eigen::Matrix3f E;
    E << e(0), e(1), e(2),
         e(3), e(4), e(5),
         e(6), e(7), e(8);

    // 强制 rank = 2
    Eigen::JacobiSVD<Eigen::Matrix3f> svdE(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3f singularValues = svdE.singularValues(); // 3个奇异值自动从大到小排列
    float avgSigma = (singularValues(0) + singularValues(1)) / 2.0f;

    Eigen::Matrix3f correctedSigma = Eigen::Matrix3f::Zero();
    correctedSigma(0, 0) = avgSigma;
    correctedSigma(1, 1) = avgSigma;
    correctedSigma(2, 2) = 0.0f;

    Eigen::Matrix3f Erefined = svdE.matrixU() * correctedSigma * svdE.matrixV().transpose();
    return Erefined;
}