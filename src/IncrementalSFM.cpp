// src/IncrementalSFM.cpp
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

struct CameraPose 
{
    Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
    Eigen::Vector3f t = Eigen::Vector3f::Zero();
};

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

std::vector<ColoredPoint3D> runIncrementalSFM(const CameraIntrinsics& intr, int& imagesProcessed)
{
    std::vector<ColoredPoint3D> pointCloud;
    std::unordered_map<long long, int> track;

    imagesProcessed = 0;

    std::vector<cv::Mat> images;
    int idx = 1;
    while (true) 
    {
        cv::Mat img = robustImreadIndexed(idx);
        if (img.empty()) break;
        images.push_back(img);
        idx++;
    }

    const int n = static_cast<int>(images.size());
    if (n < 2) {
        std::cerr << "[IncrementalSFM] At least 2 images are required, but only found "
            << n << std::endl;
        return pointCloud;
    }
    std::cout << "[IncrementalSFM] Found "
        << n
        << " images, starting incremental reconstruction"
        << std::endl;

    std::vector<FeatureSet> features(n);
    for (int i = 0; i < n; ++i) {
        features[i] = extractSIFTFeatures(images[i]);
    }
    std::vector<CameraPose> cameras(n);

    // ==================== 初始两视图重建(img0, img1) ====================
    std::vector<Match> matches01 = matchBruteForce(features[0], features[1], 0.75f);
    std::cout << "img0-img1 matched "
        << matches01.size()
        << " feature pairs"
        << std::endl;

    NormalizedCorrespondences corr01 = buildNormalizedCorrespondences(features[0], features[1], matches01, intr, intr);
    RansacResult ransac01 = ransacEssentialMatrix(corr01.pts1, corr01.pts2, 1.5f, intr.fx);
    std::cout << "[IncrementalSFM] img0-img1 inliers: "
        << ransac01.inlierIndices.size()
        << " / "
        << matches01.size()
        << std::endl;

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
        std::cerr << "[IncrementalSFM] Failed to recover pose between img0 and img1. Cannot continue."
            << std::endl;
        return pointCloud;
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
        initPassed++;
    }
    std::cout << "[IncrementalSFM] Initial two-view triangulation: "
        << initPassed
        << " / "
        << ransac01.inlierIndices.size()
        << " points passed cheirality check"
        << std::endl;

    imagesProcessed = 2;

    // ==================== 链式扩展第3张及以后的图片 ====================
    int lastRegisteredIdx = 1;

    for (int i = 2; i < n; ++i) {
        int prevIdx = lastRegisteredIdx;

        std::vector<Match> matchesPrev = matchBruteForce(features[prevIdx], features[i], 0.75f);
        std::cout << "\nimg"
            << prevIdx
            << "-img"
            << i
            << " matched "
            << matchesPrev.size()
            << " feature pairs"
            << std::endl;

        std::vector<Eigen::Vector3f> pnpObjectPoints;
        std::vector<Eigen::Vector2f> pnpImagePoints;
        std::vector<int> newMatchIndices;

        for (int m = 0; m < static_cast<int>(matchesPrev.size()); ++m) {
            int kpPrev = matchesPrev[m].idx1;
            auto it = track.find(makeTrackKey(prevIdx, kpPrev));
            if (it != track.end()) {
                pnpObjectPoints.push_back(pointCloud[it->second].position);
                int kpCur = matchesPrev[m].idx2;
                pnpImagePoints.emplace_back(features[i].kpX[kpCur], features[i].kpY[kpCur]);
            }
            else {
                newMatchIndices.push_back(m);
            }
        }

        std::cout << "3D-2D correspondences for PnP: "
            << pnpObjectPoints.size()
            << ", potential new point candidates: "
            << newMatchIndices.size()
            << std::endl;

        PnPResult pnpResult = solvePnPForNewView(pnpObjectPoints, pnpImagePoints, intr);
        if (!pnpResult.success) {
            std::cerr << "[IncrementalSFM] PnP failed for img"
                << i
                << ", skipping this view"
                << std::endl;
            continue;
        }

        cameras[i].R = pnpResult.R;
        cameras[i].t = pnpResult.t;
        imagesProcessed++;
        lastRegisteredIdx = i;

        std::cout << "[IncrementalSFM] img"
            << i
            << " pose recovered successfully"
            << std::endl;

        ProjectionMatrix Pprev;
        Pprev.block<3, 3>(0, 0) = cameras[prevIdx].R;
        Pprev.block<3, 1>(0, 3) = cameras[prevIdx].t;
        ProjectionMatrix Pcur;
        Pcur.block<3, 3>(0, 0) = cameras[i].R;
        Pcur.block<3, 1>(0, 3) = cameras[i].t;

        NormalizedCorrespondences corrPrevCur = buildNormalizedCorrespondences(features[prevIdx], features[i], matchesPrev, intr, intr);

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
            newPassed++;
        }

        std::cout << "[IncrementalSFM] img"
            << i
            << " added "
            << newPassed
            << " new triangulated points, current point cloud size: "
            << pointCloud.size()
            << std::endl;
    }

    return pointCloud;
}