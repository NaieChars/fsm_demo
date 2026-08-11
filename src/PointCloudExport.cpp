#include "PointCloudExport.h"
#include <fstream>
#include <iostream>

bool savePointCloudPLY(const std::vector<ColoredPoint3D>& points, const std::string& path)
{
    std::ofstream ofs(path);
    if (!ofs.is_open()) 
    {
        std::cerr << "[PointCloudExport] Failed to open file for writing: " << path << std::endl;
        return false;
    }

    ofs << "ply\n";
    ofs << "format ascii 1.0\n";
    ofs << "element vertex " << points.size() << "\n";
    ofs << "property float x\n";
    ofs << "property float y\n";
    ofs << "property float z\n";
    ofs << "property uchar red\n";
    ofs << "property uchar green\n";
    ofs << "property uchar blue\n";
    ofs << "end_header\n";

    for (const auto& p : points) 
    {
        ofs << p.position.x() << " " << p.position.y() << " " << p.position.z() << " "
            << static_cast<int>(p.r) << " " << static_cast<int>(p.g) << " " << static_cast<int>(p.b) << "\n";
    }

    ofs.close();
    std::cout << "[PointCloudExport] Saved point cloud into: " << path << " (Total " << points.size() << "points)" << std::endl;
    return true;
}
