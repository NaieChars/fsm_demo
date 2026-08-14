// src/IncrementalSFM.cpp (完整内容，替换你现有文件)
#include "IncrementalSFM.h"
#include "FeatureSet.h"
#include "FeatureExtractor.h"
#include "FeatureMatcher.h"
#include "Correspondences.h"
#include "RANSAC.h"
#include "PoseRecovery.h"
#include "Triangulation.h"
#include "PnPSolver.h"
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <unordered_map>
#include <algorithm>
#include <iostream>
#include <string>

static long long makeTrackKey(int imageIdx, int kpIdx)
{
    return (static_cast<long long>(imageIdx) << 32) | static_cast<unsigned int>(kpIdx);
}

static cv::Mat robustImreadIndexed(int index)
{
    std::string filename = "data/img" + std::to_string(index) + ".jpg";
    std::vector<std::string> candidates = {
        filename, "../" + filename, "../../" + filename, "../../../" + filename
    };
    for (const auto& path : candidates) {
        cv::Mat img = cv::imread(path);
        if (!img.empty()) return img;
    }
    return cv::Mat();
}

static void sampleColor(const FeatureSet& f, const cv::Mat& img, int kpIdx,
    unsigned char& r, unsigned char& g, unsigned char& b)
{
    int px = std::clamp(static_cast<int>(f.kpX[kpIdx]), 0, img.cols - 1);
    int py = std::clamp(static_cast<int>(f.kpY[kpIdx]), 0, img.rows - 1);
    cv::Vec3b bgr = img.at<cv::Vec3b>(py, px);
    r = bgr[2]; g = bgr[1]; b = bgr[0];
}

SFMResult runIncrementalSFM(const CameraIntrinsics& intr, int& imagesProcessed)
{
    SFMResult result;
    std::unordered_map<long long, int> track;

    imagesProcessed = 0;

    std::vector<cv::Mat> images;
    int idx = 1;
    while (true) {
        cv::Mat img = robustImreadIndexed(idx);
        if (img.empty()) break;
        images.push_back(img);
        idx++;
    }

    const int n = static_cast<int>(images.size());
    if (n < 2) {
        std::cerr << "[IncrementalSFM] 至少需要2张图片，实际找到 " << n << " 张" << std::endl;
        return result;
    }
    std::cout << "[IncrementalSFM] 共找到 " << n << " 张图片，开始增量重建" << std::endl;

    std::vector<FeatureSet> features(n);
    for (int i = 0; i < n; ++i) {
        features[i] = extractSIFTFeatures(images[i]);
    }

    auto& pointCloud = result.pointCloud;
    auto& observations = result.observations;
    result.cameras.resize(n);
    auto& cameras = result.cameras;

    // ==================== 初始两视图重建(img0, img1) ====================
    std::vector<Match> matches01 = matchBruteForce(features[0], features[1], 0.75f);
    std::cout << "img0-img1 共匹配到 " << matches01.size() << " 对特征点" << std::endl;

    NormalizedCorrespondences corr01 = buildNormalizedCorrespondences(features[0], features[1], matches01, intr, intr);
    RansacResult ransac01 = ransacEssentialMatrix(corr01.pts1, corr01.pts2, 1.5f, intr.fx);
    std::cout << "[IncrementalSFM] img0-img1 内点数: " << ransac01.inlierIndices.size()
        << " / " << matches01.size() << std::endl;

    std::vector<Eigen::Vector2f> inlierPts0, inlierPts1;
    inlierPts0.reserve(ransac01.inlierIndices.size());
    inlierPts1.reserve(ransac01.inlierIndices.size());
    for (int i : ransac01.inlierIndices) {
        inlierPts0.push_back(corr01.pts1[i]);
        inlierPts1.push_back(corr01.pts2[i]);
    }

    cameras[0].R = Eigen::Matrix3f::Identity();
    cameras[0].t = Eigen::Vector3f::Zero();
    if (!recoverPose(ransac01.E, inlierPts0, inlierPts1, cameras[1].R, cameras[1].t)) {
        std::cerr << "[IncrementalSFM] img0-img1 位姿恢复失败，无法继续" << std::endl;
        return result;
    }

    ProjectionMatrix P0;
    P0 << Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero();
    ProjectionMatrix P1;
    P1.block<3, 3>(0, 0) = cameras[1].R;
    P1.block<3, 1>(0, 3) = cameras[1].t;

    int initPassed = 0;
    for (size_t k = 0; k < ransac01.inlierIndices.size(); ++k) {
        int matchIdx = ransac01.inlierIndices[k];
        Eigen::Vector3f X = triangulatePointDLT(P0, P1, inlierPts0[k], inlierPts1[k]);

        float depth0 = X.z();
        float depth1 = (cameras[1].R * X + cameras[1].t).z();
        if (depth0 <= 0.0f || depth1 <= 0.0f) continue;

        int kp0 = matches01[matchIdx].idx1;
        int kp1 = matches01[matchIdx].idx2;

        ColoredPoint3D cp;
        cp.position = X;
        sampleColor(features[0], images[0], kp0, cp.r, cp.g, cp.b);
        pointCloud.push_back(cp);

        int landmarkIdx = static_cast<int>(pointCloud.size()) - 1;
        track[makeTrackKey(0, kp0)] = landmarkIdx;
        track[makeTrackKey(1, kp1)] = landmarkIdx;

        observations.push_back({ 0, landmarkIdx, inlierPts0[k] });
        observations.push_back({ 1, landmarkIdx, inlierPts1[k] });
        initPassed++;
    }
    std::cout << "[IncrementalSFM] 初始两视图三角化: " << initPassed << " / "
        << ransac01.inlierIndices.size() << " 个点通过cheirality检验" << std::endl;

    imagesProcessed = 2;

    // ==================== 链式扩展第3张及以后的图片 ====================
    int lastRegisteredIdx = 1;

    for (int i = 2; i < n; ++i) {
        int prevIdx = lastRegisteredIdx;

        std::vector<Match> matchesPrev = matchBruteForce(features[prevIdx], features[i], 0.75f);
        std::cout << "\nimg" << prevIdx << "-img" << i << " 共匹配到 " << matchesPrev.size() << " 对特征点" << std::endl;

        NormalizedCorrespondences corrPrevCur = buildNormalizedCorrespondences(features[prevIdx], features[i], matchesPrev, intr, intr);

        std::vector<Eigen::Vector3f> pnpObjectPoints;
        std::vector<Eigen::Vector2f> pnpImagePoints;
        std::vector<int> newMatchIndices;

        struct PendingOldObs { int landmarkIdx; Eigen::Vector2f uv; };
        std::vector<PendingOldObs> pendingOldObs;

        for (int m = 0; m < static_cast<int>(matchesPrev.size()); ++m) {
            int kpPrev = matchesPrev[m].idx1;
            auto it = track.find(makeTrackKey(prevIdx, kpPrev));
            if (it != track.end()) {
                int landmarkIdx = it->second;
                pnpObjectPoints.push_back(pointCloud[landmarkIdx].position);
                int kpCur = matchesPrev[m].idx2;
                pnpImagePoints.emplace_back(features[i].kpX[kpCur], features[i].kpY[kpCur]);
                pendingOldObs.push_back({ landmarkIdx, corrPrevCur.pts2[m] });
            }
            else {
                newMatchIndices.push_back(m);
            }
        }

        std::cout << "其中可用于PnP的3D-2D对应点: " << pnpObjectPoints.size()
            << "，潜在的全新点候选: " << newMatchIndices.size() << std::endl;

        PnPResult pnpResult = solvePnPForNewView(pnpObjectPoints, pnpImagePoints, intr);
        if (!pnpResult.success) {
            std::cerr << "[IncrementalSFM] img" << i << " PnP求解失败，跳过该视角" << std::endl;
            continue;
        }

        cameras[i].R = pnpResult.R;
        cameras[i].t = pnpResult.t;
        imagesProcessed++;
        lastRegisteredIdx = i;

        std::cout << "[IncrementalSFM] img" << i << " 位姿恢复成功" << std::endl;

        for (const auto& po : pendingOldObs) {
            observations.push_back({ i, po.landmarkIdx, po.uv });
        }

        ProjectionMatrix Pprev;
        Pprev.block<3, 3>(0, 0) = cameras[prevIdx].R;
        Pprev.block<3, 1>(0, 3) = cameras[prevIdx].t;
        ProjectionMatrix Pcur;
        Pcur.block<3, 3>(0, 0) = cameras[i].R;
        Pcur.block<3, 1>(0, 3) = cameras[i].t;

        int newPassed = 0;
        for (int m : newMatchIndices) {
            Eigen::Vector3f X = triangulatePointDLT(Pprev, Pcur, corrPrevCur.pts1[m], corrPrevCur.pts2[m]);

            float depthPrev = (cameras[prevIdx].R * X + cameras[prevIdx].t).z();
            float depthCur = (cameras[i].R * X + cameras[i].t).z();
            if (depthPrev <= 0.0f || depthCur <= 0.0f) continue;

            int kpPrev = matchesPrev[m].idx1;
            int kpCur = matchesPrev[m].idx2;

            ColoredPoint3D cp;
            cp.position = X;
            sampleColor(features[prevIdx], images[prevIdx], kpPrev, cp.r, cp.g, cp.b);
            pointCloud.push_back(cp);

            int landmarkIdx = static_cast<int>(pointCloud.size()) - 1;
            track[makeTrackKey(prevIdx, kpPrev)] = landmarkIdx;
            track[makeTrackKey(i, kpCur)] = landmarkIdx;

            observations.push_back({ prevIdx, landmarkIdx, corrPrevCur.pts1[m] });
            observations.push_back({ i, landmarkIdx, corrPrevCur.pts2[m] });
            newPassed++;
        }

        std::cout << "[IncrementalSFM] img" << i << " 新增 " << newPassed << " 个三角化点，当前点云总数: "
            << pointCloud.size() << std::endl;
    }

    return result;
}