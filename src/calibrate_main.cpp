#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include "CameraCalibration.h"
#include "CameraIntrinsics.h"

namespace fs = std::filesystem;

int main(int argc, char** argv)
{
    if (argc < 2) 
    {
        std::cerr << "用法: calibrate_tool.exe <squareSizeMM>" << std::endl;
        std::cerr << "  squareSizeMM: 棋盘格每格的实测物理边长(毫米), 例如: calibrate_tool.exe 25.4" << std::endl;
        return -1;
    }

    float squareSizeMM = std::stof(argv[1]);

    // 内部角点数：对应 generate_checkerboard.cpp 里生成的 10x7 格子棋盘
    cv::Size boardSize(9, 6);

    // 扫描 data/calib/ 文件夹下所有图片文件
    std::vector<std::string> calibDirs = { "data/calib", "../data/calib", "../../data/calib", "../../../data/calib" };
    std::string foundDir;
    for (const auto& dir : calibDirs) 
    {
        if (fs::exists(dir) && fs::is_directory(dir)) 
        {
            foundDir = dir;
            break;
        }
    }

    if (foundDir.empty()) 
    {
        std::cerr << "未找到 data/calib 文件夹，请确认标定照片已放入该目录" << std::endl;
        return -1;
    }

    std::vector<std::string> imagePaths;
    for (const auto& entry : fs::directory_iterator(foundDir)) 
    {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".JPG" || ext == ".PNG") 
        {
            imagePaths.push_back(entry.path().string());
        }
    }

    std::cout << "在 " << foundDir << " 中找到 " << imagePaths.size() << " 张图片" << std::endl;

    if (imagePaths.empty()) 
    {
        std::cerr << "文件夹内没有图片文件" << std::endl;
        return -1;
    }

    CameraIntrinsics intr;
    double rmsError = runCameraCalibration(imagePaths, boardSize, squareSizeMM, intr);

    if (rmsError < 0.0) 
    {
        std::cerr << "标定失败" << std::endl;
        return -1;
    }

    std::cout << "\n===== 标定结果 =====" << std::endl;
    std::cout << "RMS重投影误差: " << rmsError << " 像素" << std::endl;
    std::cout << "fx = " << intr.fx << ", fy = " << intr.fy << std::endl;
    std::cout << "cx = " << intr.cx << ", cy = " << intr.cy << std::endl;
    std::cout << "畸变系数 (k1,k2,p1,p2,k3) = "
        << intr.distCoeffs[0] << ", " << intr.distCoeffs[1] << ", "
        << intr.distCoeffs[2] << ", " << intr.distCoeffs[3] << ", "
        << intr.distCoeffs[4] << std::endl;

    saveIntrinsicsToFile(intr, "camera_intrinsics.txt");

    return 0;
}