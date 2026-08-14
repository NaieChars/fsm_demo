cmake_minimum_required(VERSION 3.20)
project(sfm_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OpenCV_DIR "C:/Dev/opencv/build")
set(EIGEN3_INCLUDE_DIR "C:/Dev/eigen-5.0.1")
# ============================================

find_package(OpenCV REQUIRED)

# ---- 主程序: 完整SFM流程 ----
add_executable(sfm_demo
    src/main.cpp
    src/FeatureExtractor.cpp
    src/FeatureMatcher.cpp
    src/Visualization.cpp
    src/CameraIntrinsics.cpp
    src/Correspondences.cpp
    src/EssentialMatrix.cpp
    src/RANSAC.cpp
    src/PoseRecovery.cpp
    src/Triangulation.cpp
    src/PointCloudExport.cpp
    src/PnPSolver.cpp
    src/IncrementalSFM.cpp
    src/BundleAdjustment.cpp
)
target_include_directories(sfm_demo PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    src
)
target_link_libraries(sfm_demo PRIVATE ${OpenCV_LIBS})

# ---- 独立工具1: 生成棋盘格图案 ----
add_executable(generate_checkerboard
    tools/generate_checkerboard.cpp
)
target_include_directories(generate_checkerboard PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(generate_checkerboard PRIVATE ${OpenCV_LIBS})

# ---- 独立工具2: 相机标定 ----
add_executable(calibrate_tool
    src/calibrate_main.cpp
    src/CameraCalibration.cpp
    src/CameraIntrinsics.cpp
)
target_include_directories(calibrate_tool PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    src
)
target_link_libraries(calibrate_tool PRIVATE ${OpenCV_LIBS})