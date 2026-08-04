cmake_minimum_required(VERSION 3.20)
project(sfm_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OpenCV_DIR "C:/Dev/opencv/build")
set(EIGEN3_INCLUDE_DIR "C:/Dev/eigen-5.0.1")
# ============================================

find_package(OpenCV REQUIRED)

add_executable(sfm_demo src/main.cpp)

target_include_directories(sfm_demo PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(sfm_demo PRIVATE ${OpenCV_LIBS})