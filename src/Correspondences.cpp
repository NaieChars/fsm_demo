#include "Correspondences.h"

NormalizedCorrespondences buildNormalizedCorrespondences(
    const FeatureSet& f1, const FeatureSet& f2,
    const std::vector<Match>& matches,
    const CameraIntrinsics& intr1, const CameraIntrinsics& intr2)
{
    NormalizedCorrespondences corr;
    corr.pts1.reserve(matches.size());
    corr.pts2.reserve(matches.size());

    for (const auto& m : matches) 
    {
        Eigen::Vector2f p1 = pixelToNormalized(intr1, f1.kpX[m.idx1], f1.kpY[m.idx1]);
        Eigen::Vector2f p2 = pixelToNormalized(intr2, f2.kpX[m.idx2], f2.kpY[m.idx2]);
        corr.pts1.push_back(p1);
        corr.pts2.push_back(p2);
    }

    return corr;
}