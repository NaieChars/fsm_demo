# sfm_demo

一个从零手写的增量式 Structure-from-Motion (SfM) 实现：根据一组多视角图像，恢复相机内参、每个视角的位姿，并重建出带颜色的稀疏 3D 点云。**核心几何算法**（8点法、RANSAC、位姿分解、三角化、Bundle Adjustment）均为自行实现，未依赖 g2o / Ceres / COLMAP 等现成 SfM/优化库；OpenCV 仅用于图像 I/O、SIFT 特征提取、相机标定和 PnP 求解这类工程性组件。

![C++](https://img.shields.io/badge/C++-17-blue) ![CMake](https://img.shields.io/badge/CMake-3.20+-brightgreen) ![OpenCV](https://img.shields.io/badge/OpenCV-4.x-orange) ![Eigen](https://img.shields.io/badge/Eigen-3.4+-purple)

## 效果

17 视角、约 3.3 万点的稀疏重建，Bundle Adjustment + 离群点过滤后平均重投影误差约 2 像素。（些许外点可能来自背景 HDRI 干扰）  
在 blender 里面放置模型基于物理相机进行拍摄，根据现实中标定好的相机进行参数设置，程序读取 `camera_intrinsics_blender.txt` 里的相机数据。实际拍摄17张照片（并未覆盖完整个物体是考虑到我背景用的 HDRI，没拍摄的那部分有背景干扰会增大“外点”数量）

| 阶段 | 输出 |
|---|---|
| 增量式重建（未优化） | `sparse_pointcloud_before_ba.ply` |
| Bundle Adjustment 后 | `sparse_pointcloud_after_ba.ply` |
| 离群点过滤 + 精修后 | `sparse_pointcloud_final.ply` |

用 MeshLab 打开上述 PLY 文件查看。

## 算法流程

1. **相机标定**：棋盘格标定（`cv::calibrateCamera`），得到内参矩阵 K 和畸变系数 (k1, k2, p1, p2, k3)。（后面因效果不佳换成在 blender 里面进行拍摄获取数据源，blender 物理相机没有畸变系数，`camera_intrinsics_blender.txt` 第二行为0.
2. **特征提取与匹配**：SIFT 特征点 + 暴力匹配（SoA 内存布局，考虑到后续 CUDA 移植） + Lowe's ratio test。
3. **去畸变归一化**：`cv::undistortPoints` 将像素坐标转换为去畸变的归一化相机坐标。
4. **本质矩阵估计**：归一化 8 点法（SVD 求解齐次线性方程组的零空间）+ 秩-2 约束修正，外层套 RANSAC（Sampson 距离评分，自适应迭代次数）剔除误匹配。
5. **位姿恢复**：对本质矩阵 SVD 分解出 4 组候选 (R, t)，用 cheirality check（三角化后检验点是否在两相机前方）选出正确解。
6. **三角化**：线性 DLT 法，从两组归一化坐标求解 3D 点的齐次坐标。
7. **增量式多视角扩展**：新视角通过 PnP（`cv::solvePnPRansac`）对齐到已重建的 3D 点，未关联的匹配点三角化为新的 3D 点；用全局 track 表记录跨视角的观测关系，链式匹配「最近一次成功注册的视角」而非严格按序，避免单帧失败导致整条链中断。
8. **Bundle Adjustment**：Levenberg-Marquardt 联合优化相机位姿与 3D 点坐标。旋转在 SO(3) 流形上用 Rodrigues 公式做局部扰动；固定首个相机以消除规范自由度；用 Schur 补消去点参数，只对相机块（远小于点数）求解稠密线性方程组；Huber 核函数降权离群观测，避免其在优化中主导解。
9. **离群点过滤**：BA 收敛后按像素残差阈值剔除仍然过大的观测，重新做一轮精修。

## 项目结构

```
sfm_demo/
├── CMakeLists.txt
├── data/
│   ├── calib/               # 标定用棋盘格照片
│   ├── img1.jpg             # 重建用图像，按编号连续命名
│   ├── img2.jpg
│   └── ...
├── tools/
│   └── generate_checkerboard.cpp   # 生成标定用棋盘格图案
└── src/
    ├── main.cpp                     # 主流程入口
    ├── CameraIntrinsics.h/.cpp      # 内参结构体、文件读写
    ├── CameraCalibration.h/.cpp     # 棋盘格标定
    ├── calibrate_main.cpp           # 独立标定工具入口
    ├── FeatureSet.h                 # 特征点/描述子存储（SoA 布局）
    ├── FeatureExtractor.h/.cpp      # SIFT 特征提取
    ├── FeatureMatcher.h/.cpp        # 暴力匹配 + ratio test
    ├── Correspondences.h/.cpp       # 去畸变归一化坐标构建
    ├── EssentialMatrix.h/.cpp       # 归一化 8 点法
    ├── RANSAC.h/.cpp                # 本质矩阵 RANSAC
    ├── PoseRecovery.h/.cpp          # E 矩阵分解 + cheirality check
    ├── Triangulation.h/.cpp         # DLT 三角化
    ├── PnPSolver.h/.cpp             # 新视角 PnP 位姿求解
    ├── IncrementalSFM.h/.cpp        # N 视角增量式重建主流程
    ├── BundleAdjustment.h/.cpp      # LM + Schur 补 + Huber 核 + 离群点过滤
    ├── PointCloudExport.h/.cpp      # PLY 点云导出
    ├── SFMTypes.h                   # 跨模块共享类型（CameraPose / Observation）
    └── Visualization.h/.cpp         # 匹配结果可视化（调试用）
```

## 依赖

- C++17
- CMake 3.20+
- OpenCV 4.x（`core` `imgproc` `calib3d` `features2d`）
- Eigen 3.4+

在 Windows + Visual Studio 2022 下用 OpenCV 官方预编译包和 Eigen 源码包测试通过；CMakeLists.txt 里的 `OpenCV_DIR` 和 `EIGEN3_INCLUDE_DIR` 按本机实际路径修改。

## 构建

```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

会生成三个可执行文件：

- `generate_checkerboard`：生成棋盘格标定图案
- `calibrate_tool`：相机标定
- `sfm_demo`：主流程

Debug 模式下 Eigen 的矩阵运算未做向量化优化，Bundle Adjustment 在大规模点云上会明显偏慢，建议用 Release 编译运行。

## 使用

### 1. 生成并显示棋盘格

```bash
./generate_checkerboard
```

生成 `checkerboard.png`，在屏幕上全屏显示（或打印），用尺子实测一个格子的物理边长（毫米），标定时要用。

### 2. 拍摄标定照片

用要标定的相机对着棋盘格拍 15~20 张，覆盖不同角度和画面四角，锁定对焦/变焦，存入 `data/calib/`。

后面我直接用的 python 脚本程序在 blender 里面拍摄，质量更高

### 3. 运行标定

```bash
./calibrate_tool <squareSizeMM>
```

例如 `./calibrate_tool 25.4`。成功后生成 `camera_intrinsics.txt`（内参 + 畸变系数 + 标定时的图像分辨率）。

### 4. 准备重建用图像

用与标定时相同的相机设置（焦距、变焦、分辨率）拍摄场景，按 `data/img1.jpg`, `data/img2.jpg`, ... 连续编号存放。相邻视角建议有明显平移/旋转视差，避免纹理高度重复的场景（不规则重复图案会显著拉低正确匹配的比例）。

### 5. 运行重建

```bash
./sfm_demo
```

依次完成特征匹配、位姿估计、三角化、增量式多视角扩展、Bundle Adjustment，输出三份 PLY 点云和控制台日志（匹配数、内点率、重投影误差等）。

## 已知限制

- 新视角只与最近一次成功注册的单个视角做链式匹配，没有做多视角联合匹配或回环检测，跨越较大角度时容易因重叠不足而配准失败。
- 单线程 CPU 实现，未做 SIMD/多线程优化；数据结构（SoA 布局、独立采样批处理等）在设计时已考虑后续 CUDA 移植。
- 仅稀疏重建，没有 MVS 稠密化阶段。
- 畸变模型仅覆盖径向 (k1, k2, k3) 和切向 (p1, p2) 畸变，不支持鱼眼等大畸变镜头。

## 后续计划

- 关键计算模块（特征匹配、RANSAC）的 CUDA 并行化
- 基于 PatchMatch 的稠密 MVS 重建
- 多摄像头节点的分布式同步采集
