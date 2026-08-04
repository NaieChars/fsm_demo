#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <Windows.h>


int main()
{
    SetConsoleCP(CP_UTF8); 
    SetConsoleOutputCP(CP_UTF8);
    // ---- 验证 OpenCV ----
    // 先不读外部图片文件，用OpenCV自己创建一张纯色测试图，
    // 避免这一步同时排查"库链接"和"文件路径"两类问题
    cv::Mat testImg(480, 640, CV_8UC3, cv::Scalar(0, 128, 255));
    std::cout << "[OpenCV] 创建图像成功, 尺寸: "
              << testImg.cols << "x" << testImg.rows
              << ", 通道数: " << testImg.channels() << std::endl;
    std::cout << "[OpenCV] 版本号: " << CV_VERSION << std::endl;

    // ---- 验证 Eigen ----
    Eigen::Matrix3f R;
    R << 1, 0, 0,
         0, 1, 0,
         0, 0, 1;
    Eigen::Vector3f t(1.0f, 2.0f, 3.0f);

    std::cout << "[Eigen] 单位矩阵 R = \n" << R << std::endl;
    std::cout << "[Eigen] 向量 t = " << t.transpose() << std::endl;
    std::cout << "[Eigen] R * t = " << (R * t).transpose() << std::endl;
    std::cout << "[Eigen] R的行列式 = " << R.determinant() << std::endl;

    // ---- 弹窗验证 OpenCV 的 highgui 模块（GUI显示）也正常 ----
    cv::imshow("Environment Check", testImg);
    std::cout << "\n窗口已弹出，按任意键关闭窗口并结束程序..." << std::endl;
    cv::waitKey(0);

    return 0;
}