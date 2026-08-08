// 独立工具，生成一张标准棋盘图案，用于相机标定

#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{
    // 棋盘格参数：注意 OpenCV 的 findChessboardCorners 检测的是"内部角点"数量，
    // 不是格子数量。比如 10x7 个格子，内部角点是 9x6 个（每个方向少1）。
    // 这里先按 10x7 格子生成图案，后续标定代码里用 (9,6) 作为内部角点数。
    const int squaresX = 10;      // 横向格子数
    const int squaresY = 7;       // 纵向格子数
    const int squarePixels = 150; // 每个格子在图片里占多少像素（越大后续显示越清晰）
    const int borderPixels = 100; // 图片四周留白，避免棋盘格贴边导致角点检测困难

    int imgWidth = squaresX * squarePixels + 2 * borderPixels;
    int imgHeight = squaresY * squarePixels + 2 * borderPixels;

    // 整张图先填白色背景
    cv::Mat board(imgHeight, imgWidth, CV_8UC1, cv::Scalar(255));

    for (int row = 0; row < squaresY; ++row) 
    {
        for (int col = 0; col < squaresX; ++col) 
        {
            // 经典棋盘格规则：(row+col)为偶数时涂黑
            if ((row + col) % 2 == 0) 
            {
                int x0 = borderPixels + col * squarePixels;
                int y0 = borderPixels + row * squarePixels;
                cv::Rect square(x0, y0, squarePixels, squarePixels);
                board(square).setTo(cv::Scalar(0));
            }
        }
    }

    std::string outPath = "checkerboard.png";
    cv::imwrite(outPath, board);
    std::cout << "棋盘格已生成: " << outPath << std::endl;
    std::cout << "格子数(横x纵): " << squaresX << "x" << squaresY << std::endl;
    std::cout << "对应的内部角点数(标定时使用): " << (squaresX - 1) << "x" << (squaresY - 1) << std::endl;

    return 0;
}