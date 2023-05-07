#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/ximgproc/edge_filter.hpp>
#include <fstream>
#include <sstream>
#include <string> 
#include <thread>

#define THREADS_PER_BLOCK 8

typedef uint8_t UCHAR;

// GPU Device implemented functions

__global__
void Bilateral_CUDA(UCHAR * I, UCHAR * R, 
               int width, int height, int win_size,
               UCHAR * spatialMask, float * rangeMask);

__global__
void JB_CUDA(UCHAR * F, UCHAR * G, UCHAR * R, 
               int width, int height, int win_size,
               UCHAR * spatialMask, float * rangeMask);

__global__
void JBU_CUDA(UCHAR * F, UCHAR * G, UCHAR * R, 
               int width_hi, int height_hi,
               int width_lo, int height_lo, 
               int win_size,
               UCHAR * spatialMask, float * rangeMask);

// Host Device implemented functions

__host__
void CreateGaussianMask(const int& window_size, cv::Mat& mask, int& sum_mask, const double sigma);

__host__
void CreateGaussianrange(float* range_mask, float sigmaRange);

__host__
void JBU_Filtering(cv::Mat& low_res_, cv::Mat& high_res_, cv::Mat& Result, int& window_size, double sigma_spatial, double sigma_range);

__host__
void Bilateral_Filtering(cv::Mat& img, cv::Mat& Result, int& window_size, double sigma_spatial, double sigma_range);

__host__
void JB_Filtering(cv::Mat& src, cv::Mat& guide, cv::Mat& Result, int& window_size, double sigma_spatial, double sigma_range);

__host__
void upsample(cv::Mat& low_res, cv::Mat& high_res, cv::Mat& Result, int& window_size, double sigma_spatial, double sigma_range);

__host__
void Disparity2PointCloud(
  const std::string& output_file,
  int height, int width, cv::Mat& disparities,
  const int& window_size,
  const int& dmin, const double& baseline, const double& focal_length);