#pragma once
#include <vector>
#include <string>
#include <Eigen/Dense>

struct ColoredPoint3D 
{
    Eigen::Vector3f position;
    unsigned char r, g, b;
};

// 把彩色点云保存成PLY格式文件(ascii文本格式)
bool savePointCloudPLY(const std::vector<ColoredPoint3D>& points, const std::string& path);