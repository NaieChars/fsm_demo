#include "BundleAdjustment.h"
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include <iostream>

// ---- 反对称矩阵(叉乘矩阵) ----
static Eigen::Matrix3f skewSymmetric(const Eigen::Vector3f& v)
{
    Eigen::Matrix3f S;
    S << 0.0f, -v.z(), v.y(),
        v.z(), 0.0f, -v.x(),
        -v.y(), v.x(), 0.0f;
    return S;
}

// ---- SO(3)指数映射(Rodrigues公式)：把切空间的小角度向量w，转换成对应的旋转矩阵 ----
static Eigen::Matrix3f expSO3(const Eigen::Vector3f& w)
{
    float theta = w.norm();
    if (theta < 1e-8f) {
        return Eigen::Matrix3f::Identity() + skewSymmetric(w);
    }
    Eigen::Vector3f axis = w / theta;
    Eigen::Matrix3f K = skewSymmetric(axis);
    return Eigen::Matrix3f::Identity() + std::sin(theta) * K + (1.0f - std::cos(theta)) * (K * K);
}

struct ObsJacobian {
    Eigen::Vector2f residual;
    Eigen::Matrix<float, 2, 3> Jp;
    Eigen::Matrix<float, 2, 6> Jc;
};

// 计算单条观测的原始(未加权)残差和雅可比
static ObsJacobian computeObsJacobian(const CameraPose& cam, const Eigen::Vector3f& X, const Eigen::Vector2f& uv)
{
    Eigen::Vector3f Xc = cam.R * X + cam.t;
    float invZ = 1.0f / Xc.z();
    Eigen::Vector2f proj(Xc.x() * invZ, Xc.y() * invZ);

    ObsJacobian result;
    result.residual = uv - proj;

    Eigen::Matrix<float, 2, 3> dProj_dXc;
    dProj_dXc << invZ, 0.0f, -Xc.x() * invZ * invZ,
        0.0f, invZ, -Xc.y() * invZ * invZ;
    Eigen::Matrix<float, 2, 3> dRes_dXc = -dProj_dXc;

    result.Jp = dRes_dXc * cam.R;

    Eigen::Matrix<float, 2, 3> Jt = dRes_dXc;

    Eigen::Vector3f Y = Xc - cam.t;
    Eigen::Matrix<float, 2, 3> Jr = dRes_dXc * (-skewSymmetric(Y));

    result.Jc.block<2, 3>(0, 0) = Jr;
    result.Jc.block<2, 3>(0, 3) = Jt;

    return result;
}

// ---- Huber权重：残差在阈值内权重为1，超过阈值按 delta/|r| 衰减 ----
// 用法是IRLS(迭代重加权最小二乘)：把这个权重的平方根乘到Jp/Jc/residual上，
// 后面所有J^T*J、J^T*r的累加会自动变成"加权版"，不需要改动其余任何代码
static float huberWeight(float residualNorm, float delta)
{
    if (residualNorm <= delta) return 1.0f;
    return delta / std::max(residualNorm, 1e-8f);
}

// ---- 原始(未加权)总代价，用于对外报告"平均重投影误差(像素)"这个直观数值 ----
static double computeTotalCost(const std::vector<CameraPose>& cameras,
    const std::vector<Eigen::Vector3f>& points,
    const std::vector<Observation>& observations)
{
    double totalCost = 0.0;
    for (const auto& obs : observations) {
        const CameraPose& cam = cameras[obs.cameraIdx];
        const Eigen::Vector3f& X = points[obs.pointIdx];
        Eigen::Vector3f Xc = cam.R * X + cam.t;
        Eigen::Vector2f proj(Xc.x() / Xc.z(), Xc.y() / Xc.z());
        Eigen::Vector2f r = obs.uv - proj;
        totalCost += r.squaredNorm();
    }
    return totalCost;
}

// ---- Huber总代价，用于LM内部判断"这一步更新是否该被接受"----
// 残差在阈值内按0.5*r^2算，超过阈值后只按线性增长，离群点不再能主导这个数值
static double computeTotalHuberCost(const std::vector<CameraPose>& cameras,
    const std::vector<Eigen::Vector3f>& points,
    const std::vector<Observation>& observations,
    double huberDelta)
{
    double total = 0.0;
    for (const auto& obs : observations) {
        const CameraPose& cam = cameras[obs.cameraIdx];
        const Eigen::Vector3f& X = points[obs.pointIdx];
        Eigen::Vector3f Xc = cam.R * X + cam.t;
        Eigen::Vector2f proj(Xc.x() / Xc.z(), Xc.y() / Xc.z());
        double rn = static_cast<double>((obs.uv - proj).norm());
        if (rn <= huberDelta) {
            total += 0.5 * rn * rn;
        }
        else {
            total += huberDelta * (rn - 0.5 * huberDelta);
        }
    }
    return total;
}

BAResult runBundleAdjustment(const std::vector<CameraPose>& initialCameras,
    const std::vector<Eigen::Vector3f>& initialPoints,
    const std::vector<Observation>& observations,
    float focalLengthPixels,
    int maxIterations,
    float huberDeltaPixels)
{
    BAResult result;
    std::vector<CameraPose> cameras = initialCameras;
    std::vector<Eigen::Vector3f> points = initialPoints;

    const int numCameras = static_cast<int>(cameras.size());
    const int numPoints = static_cast<int>(points.size());

    std::vector<int> freeCamIndex(numCameras, -1);
    int numFreeCameras = 0;
    for (int i = 1; i < numCameras; ++i) {
        freeCamIndex[i] = numFreeCameras++;
    }

    result.cameras = cameras;
    result.points = points;

    if (numFreeCameras == 0 || observations.empty()) {
        std::cerr << "[BundleAdjustment] 没有可优化的相机或观测，跳过" << std::endl;
        return result;
    }

    // Huber阈值换算成归一化坐标单位(和observations里的uv、以及Jp/Jc的单位一致)
    const float huberDeltaNorm = huberDeltaPixels / focalLengthPixels;

    std::vector<std::vector<int>> obsByPoint(numPoints);
    for (int i = 0; i < static_cast<int>(observations.size()); ++i) {
        obsByPoint[observations[i].pointIdx].push_back(i);
    }

    double initialCost = computeTotalCost(cameras, points, observations);
    double initialHuberCost = computeTotalHuberCost(cameras, points, observations, huberDeltaNorm);
    result.initialMeanReprojErrorPixels = std::sqrt(initialCost / observations.size()) * focalLengthPixels;
    std::cout << "[BundleAdjustment] 初始平均重投影误差: " << result.initialMeanReprojErrorPixels
        << " 像素 (共" << observations.size() << "个观测, " << numPoints << "个点, "
        << numFreeCameras << "个自由相机, Huber阈值=" << huberDeltaPixels << "像素)" << std::endl;

    double lambda = 1e-3;
    double currentCost = initialCost;
    double currentHuberCost = initialHuberCost;
    int iter = 0;

    for (; iter < maxIterations; ++iter) {
        const int camDim = 6 * numFreeCameras;
        Eigen::MatrixXd Hcc = Eigen::MatrixXd::Zero(camDim, camDim);
        Eigen::VectorXd gc = Eigen::VectorXd::Zero(camDim);

        std::vector<Eigen::Matrix3f> HppInvCache(numPoints, Eigen::Matrix3f::Zero());
        std::vector<Eigen::Vector3f> gpCache(numPoints, Eigen::Vector3f::Zero());
        std::vector<bool> pointHasObs(numPoints, false);

        struct CamContrib {
            int freeIdx;
            Eigen::Matrix<float, 6, 3> M;
            Eigen::Matrix<float, 2, 6> Jc;
            Eigen::Vector2f r;
        };

        for (int j = 0; j < numPoints; ++j) {
            if (obsByPoint[j].empty()) continue;
            pointHasObs[j] = true;

            Eigen::Matrix3f Hpp = Eigen::Matrix3f::Zero();
            Eigen::Vector3f gp = Eigen::Vector3f::Zero();
            std::vector<CamContrib> contribs;

            for (int oi : obsByPoint[j]) {
                const Observation& obs = observations[oi];
                const CameraPose& cam = cameras[obs.cameraIdx];
                ObsJacobian oj = computeObsJacobian(cam, points[j], obs.uv);

                // ---- Huber稳健加权：把权重的平方根乘到Jp/Jc/residual上 ----
                float w = huberWeight(oj.residual.norm(), huberDeltaNorm);
                float sqrtW = std::sqrt(w);
                oj.residual *= sqrtW;
                oj.Jp *= sqrtW;
                oj.Jc *= sqrtW;

                Hpp += oj.Jp.transpose() * oj.Jp;
                gp += oj.Jp.transpose() * oj.residual;

                int fi = freeCamIndex[obs.cameraIdx];
                if (fi >= 0) {
                    Eigen::Matrix<float, 6, 3> M = oj.Jc.transpose() * oj.Jp;
                    contribs.push_back({ fi, M, oj.Jc, oj.residual });
                }
            }

            Eigen::Matrix3f HppDamped = Hpp;
            for (int d = 0; d < 3; ++d) {
                HppDamped(d, d) += static_cast<float>(lambda) * std::max(Hpp(d, d), 1e-6f);
            }
            Eigen::Matrix3f HppInv = HppDamped.inverse();
            HppInvCache[j] = HppInv;
            gpCache[j] = gp;

            for (size_t a = 0; a < contribs.size(); ++a) {
                int fa = contribs[a].freeIdx;
                const auto& Ma = contribs[a].M;
                const auto& Jca = contribs[a].Jc;
                const auto& ra = contribs[a].r;

                Hcc.block(fa * 6, fa * 6, 6, 6) += (Jca.transpose() * Jca).cast<double>();
                gc.segment(fa * 6, 6) += (Jca.transpose() * ra).cast<double>();

                for (size_t b = 0; b < contribs.size(); ++b) {
                    int fb = contribs[b].freeIdx;
                    const auto& Mb = contribs[b].M;
                    Eigen::Matrix<float, 6, 6> coupling = Ma * HppInv * Mb.transpose();
                    Hcc.block(fa * 6, fb * 6, 6, 6) -= coupling.cast<double>();
                }
                gc.segment(fa * 6, 6) -= (Ma * HppInv * gp).cast<double>();
            }
        }

        for (int d = 0; d < camDim; ++d) {
            Hcc(d, d) += lambda * std::max(Hcc(d, d), 1e-6);
        }

        Eigen::VectorXd deltaCamera = Hcc.ldlt().solve(gc);

        std::vector<Eigen::Vector3f> deltaPoints(numPoints, Eigen::Vector3f::Zero());
        for (int j = 0; j < numPoints; ++j) {
            if (!pointHasObs[j]) continue;

            Eigen::Vector3f rhs = -gpCache[j];
            for (int oi : obsByPoint[j]) {
                int camIdx = observations[oi].cameraIdx;
                int fi = freeCamIndex[camIdx];
                if (fi < 0) continue;
                const CameraPose& cam = cameras[camIdx];
                ObsJacobian oj = computeObsJacobian(cam, points[j], observations[oi].uv);

                // 必须用和上面主循环完全一致的加权方式重新计算Jp/Jc，否则回代和消元用的不是同一套方程
                float w = huberWeight(oj.residual.norm(), huberDeltaNorm);
                float sqrtW = std::sqrt(w);
                oj.Jp *= sqrtW;
                oj.Jc *= sqrtW;

                Eigen::Matrix<float, 6, 3> M = oj.Jc.transpose() * oj.Jp;
                Eigen::Matrix<float, 6, 1> dc;
                dc << static_cast<float>(deltaCamera(fi * 6 + 0)), static_cast<float>(deltaCamera(fi * 6 + 1)),
                    static_cast<float>(deltaCamera(fi * 6 + 2)), static_cast<float>(deltaCamera(fi * 6 + 3)),
                    static_cast<float>(deltaCamera(fi * 6 + 4)), static_cast<float>(deltaCamera(fi * 6 + 5));
                rhs -= M.transpose() * dc;
            }
            deltaPoints[j] = HppInvCache[j] * rhs;
        }

        std::vector<CameraPose> candidateCameras = cameras;
        for (int i = 1; i < numCameras; ++i) {
            int fi = freeCamIndex[i];
            Eigen::Vector3f dr(static_cast<float>(deltaCamera(fi * 6 + 0)),
                static_cast<float>(deltaCamera(fi * 6 + 1)),
                static_cast<float>(deltaCamera(fi * 6 + 2)));
            Eigen::Vector3f dt(static_cast<float>(deltaCamera(fi * 6 + 3)),
                static_cast<float>(deltaCamera(fi * 6 + 4)),
                static_cast<float>(deltaCamera(fi * 6 + 5)));
            candidateCameras[i].R = expSO3(dr) * cameras[i].R;
            candidateCameras[i].t = cameras[i].t + dt;
        }

        std::vector<Eigen::Vector3f> candidatePoints = points;
        for (int j = 0; j < numPoints; ++j) {
            candidatePoints[j] = points[j] + deltaPoints[j];
        }

        double candidateCost = computeTotalCost(candidateCameras, candidatePoints, observations);
        double candidateHuberCost = computeTotalHuberCost(candidateCameras, candidatePoints, observations, huberDeltaNorm);

        // ---- LM的accept/reject用Huber代价判断，而不是原始L2代价 ----
        // 这一点很关键：如果这里改用candidateCost(原始L2)判断，就和"用Huber加权求解更新方向"自相矛盾了——
        // 求解时已经不再迁就离群点，但接受/拒绝这一步如果还按L2打分，等于评判标准又把离群点的话语权加回来了
        if (candidateHuberCost < currentHuberCost) {
            cameras = candidateCameras;
            points = candidatePoints;
            currentCost = candidateCost;
            currentHuberCost = candidateHuberCost;
            lambda = std::max(lambda * 0.5, 1e-7);
        }
        else {
            lambda = std::min(lambda * 2.0, 1e7);
        }

        double meanErrPixels = std::sqrt(currentCost / observations.size()) * focalLengthPixels;
        std::cout << "[BundleAdjustment] 第" << (iter + 1) << "轮, huberCost=" << currentHuberCost
            << ", lambda=" << lambda << ", 平均重投影误差=" << meanErrPixels << "像素" << std::endl;
    }

    result.cameras = cameras;
    result.points = points;
    result.finalMeanReprojErrorPixels = std::sqrt(currentCost / observations.size()) * focalLengthPixels;
    result.iterationsUsed = iter;

    return result;
}


FilterResult filterOutliers(const std::vector<CameraPose>& cameras,
    const std::vector<Eigen::Vector3f>& points,
    const std::vector<Observation>& observations,
    float focalLengthPixels,
    float outlierThresholdPixels)
{
    FilterResult result;
    result.pointKept.assign(points.size(), false);

    // 第一遍：按像素残差阈值筛掉明显离谱的观测
    std::vector<Observation> passedThreshold;
    passedThreshold.reserve(observations.size());
    for (const auto& obs : observations) {
        const CameraPose& cam = cameras[obs.cameraIdx];
        const Eigen::Vector3f& X = points[obs.pointIdx];
        Eigen::Vector3f Xc = cam.R * X + cam.t;
        Eigen::Vector2f proj(Xc.x() / Xc.z(), Xc.y() / Xc.z());
        float residualPixels = (obs.uv - proj).norm() * focalLengthPixels;
        if (residualPixels <= outlierThresholdPixels) {
            passedThreshold.push_back(obs);
        }
    }

    // 第二遍：统计每个点过滤后还剩几条观测，不足2条的点本身也要剔除(三角化的最低约束要求)
    std::vector<int> obsCountPerPoint(points.size(), 0);
    for (const auto& obs : passedThreshold) {
        obsCountPerPoint[obs.pointIdx]++;
    }
    for (size_t j = 0; j < points.size(); ++j) {
        result.pointKept[j] = obsCountPerPoint[j] >= 2;
    }

    // 第三遍：把指向"已被剔除的点"的观测也一并清掉
    result.filteredObservations.reserve(passedThreshold.size());
    for (const auto& obs : passedThreshold) {
        if (result.pointKept[obs.pointIdx]) {
            result.filteredObservations.push_back(obs);
        }
    }

    result.numObservationsRemoved = static_cast<int>(observations.size() - result.filteredObservations.size());
    result.numPointsRemoved = static_cast<int>(
        std::count(result.pointKept.begin(), result.pointKept.end(), false));

    return result;
}