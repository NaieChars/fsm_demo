#include "CameraCalibration.h"
#include <iostream>
#include <iomanip>

double runCameraCalibration(const std::vector<std::string>& imagePaths,
                            cv::Size boardSize,
                            float squareSizeMM,
                            CameraIntrinsics& outIntr)
{
    std::vector<std::vector<cv::Point3f>> objectPoints; // objectPoints: 每张成功检测到棋盘格的图片都对应一组棋盘自身坐标系下的3D点
    std::vector<std::vector<cv::Point2f>> imgPoints;    // imgPoints:每张图里实际检测到的角点2D像素坐标

    std::vector<cv::Point3f> objectCorners;
    for (int row = 0; row < boardSize.height; ++row) 
    {
        for (int col = 0; col < boardSize.width; ++col) 
        {
            objectCorners.emplace_back(col * squareSizeMM, row * squareSizeMM, 0.0f);
        }
    }

    cv::Size imageSize; 
    bool imageSizeSet = false;

    int successCount = 0;
    for (const auto& path : imagePaths) 
    {
        cv::Mat img = cv::imread(path);
        if (img.empty()) 
        {
            std::cerr << "[Calibration] 警告: 无法读取图片 " << path << "，跳过" << std::endl;
            continue;
        }

        if (!imageSizeSet) 
        {
            imageSize = img.size();
            imageSizeSet = true;
        }
        else if (img.size() != imageSize) 
        {
            // 标定要求所有照片分辨率一致，否则K矩阵定义就不统一了
            std::cerr << "[Calibration] 警告: " << path << " 分辨率(" << img.cols << "x" << img.rows
                << ")与第一张图(" << imageSize.width << "x" << imageSize.height
                << ")不一致，跳过此图" << std::endl;
            continue;
        }

        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> corners;
        // CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE 组合，对光照不均更鲁棒
        bool found = cv::findChessboardCorners(
            gray, boardSize, corners,
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);

        if (!found) 
        {
            std::cerr << "[Calibration] " << path << ": 未检测到棋盘格角点，跳过" << std::endl;
            continue;
        }

        // 亚像素级角点精修：findChessboardCorners给出的是粗略整数级角点，
        // cornerSubPix利用角点附近的梯度信息，把定位精度提升到亚像素级别，
        // 这一步对最终标定精度影响很大，不能省略
        cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001);
        cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1), criteria);

        objectPoints.push_back(objectCorners);
        imgPoints.push_back(corners);
        successCount++;

        std::cout << "[Calibration] " << path << ": 检测成功 (" << corners.size() << "个角点)" << std::endl;
    }

    std::cout << "[Calibration] 共 " << successCount << "/" << imagePaths.size()
        << " 张图片成功检测到棋盘格" << std::endl;

    // 标定至少需要几张不同角度的图才能求解，太少会导致方程欠约束或结果不稳定
    if (successCount < 5) {
        std::cerr << "[Calibration] 有效标定图片数量过少(<5)，标定结果不可靠，建议补拍" << std::endl;
        return -1.0;
    }

    cv::Mat cameraMatrix, distCoeffsMat;
    std::vector<cv::Mat> rvecs, tvecs;

    // cv::calibrateCamera 内部做的事：以"重投影误差最小"为目标，
    // 对cameraMatrix(K的4个自由参数)、distCoeffs(5个畸变参数)、
    // 以及每张图各自的(R,t)，联合做非线性最小二乘优化(Levenberg-Marquardt)。
    // 返回值就是优化收敛后的RMS重投影误差(单位:pixel)。
    double rmsError = cv::calibrateCamera(
        objectPoints, imgPoints, imageSize,
        cameraMatrix, distCoeffsMat, rvecs, tvecs);

    // 把cv::Mat(double类型)结果转换填入CameraIntrinsics结构(float类型)
    outIntr.fx = static_cast<float>(cameraMatrix.at<double>(0, 0));
    outIntr.fy = static_cast<float>(cameraMatrix.at<double>(1, 1));
    outIntr.cx = static_cast<float>(cameraMatrix.at<double>(0, 2));
    outIntr.cy = static_cast<float>(cameraMatrix.at<double>(1, 2));

    outIntr.K << outIntr.fx, 0.0f, outIntr.cx,
        0.0f, outIntr.fy, outIntr.cy,
        0.0f, 0.0f, 1.0f;
    outIntr.Kinv = outIntr.K.inverse();

    // distCoeffsMat 默认是8参数模型(在OpenCV里)，但我们只关心前5个(k1,k2,p1,p2,k3)，
    for (int i = 0; i < 5; ++i) 
    {
        outIntr.distCoeffs[i] = static_cast<float>(distCoeffsMat.at<double>(0, i));
    }

    outIntr.calibWidth = imageSize.width;
    outIntr.calibHeight = imageSize.height;

    // ------------------- 每张图片的重投影误差 ------------------------
    std::cout << "\n===== 每张图片重投影误差 =====" << std::endl;

    for (size_t i = 0; i < objectPoints.size(); ++i)
    {
        std::vector<cv::Point2f> reprojectedPoints;

        // 利用标定结果把3D角点重新投影回图像
        cv::projectPoints(
            objectPoints[i],
            rvecs[i],
            tvecs[i],
            cameraMatrix,
            distCoeffsMat,
            reprojectedPoints);

        // L2误差
        double err = cv::norm(imgPoints[i], reprojectedPoints, cv::NORM_L2);

        // 每个角点的 RMS
        double rms = std::sqrt(err * err / imgPoints[i].size());

        std::cout << imagePaths[i]
            << "    RMS = "
            << std::fixed << std::setprecision(3)
            << rms << " px" << std::endl;
    }

    return rmsError;
}