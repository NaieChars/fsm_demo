#include "Triangulation.h"

Eigen::Vector3f triangulatePointDLT(const ProjectionMatrix& P1, const ProjectionMatrix& P2,
    const Eigen::Vector2f& x1, const Eigen::Vector2f& x2)
{
    // DLT三角化
    Eigen::Matrix4f A;
    A.row(0) = x1.x() * P1.row(2) - P1.row(0);
    A.row(1) = x1.y() * P1.row(2) - P1.row(1);
    A.row(2) = x2.x() * P2.row(2) - P2.row(0);
    A.row(3) = x2.y() * P2.row(2) - P2.row(1);

    Eigen::JacobiSVD<Eigen::Matrix4f> svd(A, Eigen::ComputeFullV);
    Eigen::Vector4f Xh = svd.matrixV().col(3); // 最小奇异值对应的解，齐次坐标(X,Y,Z,W)

    // 齐次坐标归一化：除以W分量，得到欧式坐标
    return Eigen::Vector3f(Xh(0) / Xh(3), Xh(1) / Xh(3), Xh(2) / Xh(3));
}