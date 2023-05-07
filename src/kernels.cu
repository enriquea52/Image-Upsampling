#include "kernels.cuh"
// Stereo Matching Using Dynamic Programming Optimization

// parallelize the computatio for each pair of rows,
// so the number of threads is equal to height - 2*half_window_size

__host__
void CreateGaussianMask(const int& window_size, cv::Mat& mask, int& sum_mask, const double sigma)
{
	cv::Size mask_size(window_size, window_size);
	mask = cv::Mat(mask_size, CV_8UC1);

	const double hw = (window_size/2);
	const double sigmaSq = std::pow(sigma, 2);

	// rmax = 2.5 * sigma
	// sigma = rmax / 2.5

	for (int r = 0; r < window_size; ++r) {
		for (int c = 0; c < window_size; ++c) {
			// mask.at<uchar>(r, c) = 1; // box filter
			double r2 = std::pow(r - hw, 2) + std::pow(c - hw, 2);
			mask.at<uchar>(r, c) = 255*std::exp(-r2/(2*sigmaSq));
		}
	}

	// std::cout << mask <<std::endl;

	for (int r = 0; r < window_size; ++r) {
		for (int c = 0; c < window_size; ++c) {
			sum_mask += mask.at<uchar>(r, c);
		}
	}
}

__host__
void CreateGaussianrange(float* range_mask, float sigmaRange)
{
	const float sigmaRangeSq = sigmaRange*sigmaRange;
	// TODO: compute the range kernel's value
	for(int diff = 0; diff < 256; ++diff)
	{
		range_mask[diff] = std::exp(std::exp(-(diff*diff)/(2*sigmaRangeSq)));
	}
}

__global__
void Bilateral_CUDA(UCHAR * I, UCHAR * R, 
               int width, int height, int win_size,
               UCHAR * spatialMask, float * rangeMask)
{
   int half_win_size = win_size/2;
   int ROW = blockIdx.y*blockDim.y+threadIdx.y; ROW += half_win_size;
   int COL = blockIdx.x*blockDim.x+threadIdx.x; COL += half_win_size;

   int intensity_center = static_cast<int>(I[(ROW * width) + COL]);
   int intensity; float weight_range; int weight_spatial; float weight;
   int sum = 0;
   int diff;
   float sum_Bilateral_mask = 0;

   if ((ROW < height - half_win_size) && (COL < width - half_win_size))
   {
        for (int i = -half_win_size; i <= half_win_size; i++)
        {
            for (int j = -half_win_size; j <= half_win_size; j++)
            {
                intensity = static_cast<int>(I[(ROW + i) * width + (COL+j)]);
                // compute range difference to center pixxel values
                diff = std::abs(intensity_center - intensity); // ranges from 0 to 255
                // Compute the range kernel's value
                weight_range = rangeMask[diff]; 
                // Compute the spatial kernel's value
                weight_spatial = static_cast<int>(spatialMask[(i + half_win_size) * win_size + (j + half_win_size)]);
                // Combine the weights
                weight = weight_range * weight_spatial;

                sum += intensity * weight;  // covolution happening ...
                sum_Bilateral_mask += weight;
            }
        }

        R[(ROW * width) + COL] = sum / sum_Bilateral_mask;
   }

}


__global__
void JB_CUDA(UCHAR * F, UCHAR * G, UCHAR * R, 
               int width, int height, int win_size,
               UCHAR * spatialMask, float * rangeMask)
{
   int half_win_size = win_size/2;
   int ROW = blockIdx.y*blockDim.y+threadIdx.y; ROW += half_win_size;
   int COL = blockIdx.x*blockDim.x+threadIdx.x; COL += half_win_size;

   int f_center = static_cast<int>(F[(ROW * width) + COL]);
   int f_intensity, g_intensity; 
   float weight_range; int weight_spatial; float weight;
   int sum = 0;
   int diff;
   float sum_Bilateral_mask = 0;

   if ((ROW < height - half_win_size) && (COL < width - half_win_size))
   {
        for (int i = -half_win_size; i <= half_win_size; i++)
        {
            for (int j = -half_win_size; j <= half_win_size; j++)
            {
                f_intensity = static_cast<int>(F[(ROW + i) * width + (COL+j)]);
                g_intensity = static_cast<int>(G[(ROW + i) * width + (COL+j)]);

                // compute range difference to center pixxel values
                diff = std::abs(f_center - f_intensity); // ranges from 0 to 255
                // Compute the range kernel's value
                weight_range = rangeMask[diff]; 
                // Compute the spatial kernel's value
                weight_spatial = static_cast<int>(spatialMask[(i + half_win_size) * win_size + (j + half_win_size)]);
                // Combine the weights
                weight = weight_range * weight_spatial;

                sum += g_intensity * weight;  // covolution happening ...
                sum_Bilateral_mask += weight;
            }
        }

        R[(ROW * width) + COL] = sum / sum_Bilateral_mask;
   }

}


__global__
void JBU_CUDA(UCHAR * F, UCHAR * G, UCHAR * R, 
               int width_hi, int height_hi,
               int width_lo, int height_lo, 
               int win_size,
               UCHAR * spatialMask, float * rangeMask)
{
   int half_win_size = win_size/2;
   int ROW = blockIdx.y*blockDim.y+threadIdx.y; ROW += half_win_size;
   int COL = blockIdx.x*blockDim.x+threadIdx.x; COL += half_win_size;

   int f_center = static_cast<int>(F[(ROW * width_hi) + COL]);
   int f_intensity, g_intensity; 
   float weight_range; int weight_spatial; float weight;
   int sum = 0;
   int diff;
   float sum_Bilateral_mask = 0;
   int y_hi, x_hi, y_lo, x_lo;

   if ((ROW < height_hi - half_win_size) && (COL < width_hi - half_win_size))
   {
        for (int i = -half_win_size; i <= half_win_size; i++)
        {
            for (int j = -half_win_size; j <= half_win_size; j++)
            {
                y_hi = ROW + i;
                x_hi = COL + j;
                f_intensity = static_cast<int>(F[(y_hi * width_hi) + (x_hi)]);

                y_lo = std::round((y_hi/(float)height_hi)*height_lo);
                x_lo = std::round((x_hi/(float)width_hi)*width_lo);

                g_intensity = static_cast<int>(G[(y_lo * width_lo) + (x_lo)]);

                // compute range difference to center pixxel values
                diff = std::abs(f_center - f_intensity); // ranges from 0 to 255
                // Compute the range kernel's value
                weight_range = rangeMask[diff]; 
                // Compute the spatial kernel's value
                weight_spatial = static_cast<int>(spatialMask[(i + half_win_size) * win_size + (j + half_win_size)]);
                // Combine the weights
                weight = weight_range * weight_spatial;

                sum += g_intensity * weight;  // covolution happening ...
                sum_Bilateral_mask += weight;

            }
        }

        R[(ROW * width_hi) + COL] = sum / sum_Bilateral_mask;
   }

}

__host__
void Bilateral_Filtering(cv::Mat& img, cv::Mat& Result, int& window_size, double sigma_spatial, double sigma_range)
{

  // Mask window size
  int half_win_size = window_size / 2;

  // Image dimensions 
  int height = img.size().height; // number of rows
  int width = img.size().width;   // number of cols
  int size = height*width;        // Total number of pixels of input images

  // Pointer variables for both host and device computing
  UCHAR * F, * R;

  // Memory Allocation for device computing (input and output images)
  cudaMalloc((void**)&F, size*sizeof(UCHAR));
  cudaMalloc((void**)&R, size*sizeof(UCHAR));

  // Definition of Gaussian Spatial and Range Masks
  cv::Mat gaussianMask;
  int maskSum;
  float range_mask[256];

  // Definition of Sigmas
  const double sigma = sigma_spatial;
  const float sigmaRange = sigma_range;

  //  Creating spatial and range masks
  CreateGaussianMask(window_size, gaussianMask, maskSum, sigma);
  CreateGaussianrange(range_mask, sigmaRange);

  int spatialSize = gaussianMask.size().width * gaussianMask.size().height;
  int rangeSize = 256;

  UCHAR * Sm; float * Rm; // range and spatial mask

  // Allocating and copying data to device
  cudaMalloc((void**)&Sm, spatialSize*sizeof(UCHAR));
  cudaMalloc((void**)&Rm, rangeSize*sizeof(float));

  cudaMemcpy(F, img.data, size*sizeof(UCHAR), cudaMemcpyHostToDevice);                  // Original image to device
  cudaMemcpy(R, Result.data, size*sizeof(UCHAR), cudaMemcpyHostToDevice);               // result_imging image for device
  cudaMemcpy(Sm, gaussianMask.data, spatialSize*sizeof(UCHAR), cudaMemcpyHostToDevice); // spatial mask to device
  cudaMemcpy(Rm, range_mask, rangeSize*sizeof(float), cudaMemcpyHostToDevice);          // range mask to device

  int blocksNumber_x = std::ceil(width/(float)THREADS_PER_BLOCK);
  int blocksNumber_y = std::ceil(height/(float)THREADS_PER_BLOCK);

  const dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
  const dim3 blocksPerGrid(blocksNumber_x, blocksNumber_y);

  Bilateral_CUDA<<<blocksPerGrid, threadsPerBlock>>> (F, R, width, height, window_size, Sm, Rm);

  cudaMemcpy(Result.data, R, size*sizeof(UCHAR), cudaMemcpyDeviceToHost);

  // Free memory 
  cudaFree(F); cudaFree(R);
  cudaFree(Sm); cudaFree(Rm);
}

__host__
void JB_Filtering(cv::Mat& src, cv::Mat& guide, cv::Mat& Result, int& window_size, double sigma_spatial, double sigma_range)
{

  // Mask window size
  int half_win_size = window_size / 2;

  // Image dimensions 
  int height = guide.size().height; // number of rows
  int width = guide.size().width;   // number of cols
  int size = height*width;          // Total number of pixels of input images

  // Pointer variables for both host and device computing
  UCHAR * F, *G, * R;

  // Memory Allocation for device computing (input and output images)
  cudaMalloc((void**)&F, size*sizeof(UCHAR));
  cudaMalloc((void**)&G, size*sizeof(UCHAR));
  cudaMalloc((void**)&R, size*sizeof(UCHAR));

  // Definition of Gaussian Spatial and Range Masks
  cv::Mat gaussianMask;
  int maskSum;
  float range_mask[256];

  // Definition of Sigmas
  const double sigma = sigma_spatial;
  const float sigmaRange = sigma_range;

  //  Creating spatial and range masks
  CreateGaussianMask(window_size, gaussianMask, maskSum, sigma);
  CreateGaussianrange(range_mask, sigmaRange);

  int spatialSize = gaussianMask.size().width * gaussianMask.size().height;
  int rangeSize = 256;

  UCHAR * Sm; float * Rm; // range and spatial mask

  // Allocating and copying data to device
  cudaMalloc((void**)&Sm, spatialSize*sizeof(UCHAR));
  cudaMalloc((void**)&Rm, rangeSize*sizeof(float));

  cudaMemcpy(G, src.data, size*sizeof(UCHAR), cudaMemcpyHostToDevice);                  // result_imging image for device
  cudaMemcpy(F, guide.data, size*sizeof(UCHAR), cudaMemcpyHostToDevice);                // Original image to device
  cudaMemcpy(R, Result.data, size*sizeof(UCHAR), cudaMemcpyHostToDevice);               // result_imging image for device
  cudaMemcpy(Sm, gaussianMask.data, spatialSize*sizeof(UCHAR), cudaMemcpyHostToDevice); // spatial mask to device
  cudaMemcpy(Rm, range_mask, rangeSize*sizeof(float), cudaMemcpyHostToDevice);          // range mask to device

  int blocksNumber_x = std::ceil(width/(float)THREADS_PER_BLOCK);
  int blocksNumber_y = std::ceil(height/(float)THREADS_PER_BLOCK);

  const dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
  const dim3 blocksPerGrid(blocksNumber_x, blocksNumber_y);

  JB_CUDA<<<blocksPerGrid, threadsPerBlock>>>(F, G, R, width, height, window_size, Sm, Rm);

  cudaMemcpy(Result.data, R, size*sizeof(UCHAR), cudaMemcpyDeviceToHost);

  // Free memory 
  cudaFree(F); cudaFree(G); cudaFree(R);
  cudaFree(Sm); cudaFree(Rm);
}

__host__
void JBU_Filtering(cv::Mat& low_res_, cv::Mat& high_res_, cv::Mat& Result, int& window_size, double sigma_spatial, double sigma_range)
{

  cv::Mat low_res = low_res_.clone();
  cv::Mat high_res = high_res_.clone();

  // Mask window size
  int half_win_size = window_size / 2;

  // Image dimensions 
  int height_low = low_res.size().height; // number of rows
  int width_low = low_res.size().width;   // number of cols
  int size_low = height_low*width_low;    // Total number of pixels of input images

  // Image dimensions 
  int height_hi = high_res.size().height; // number of rows
  int width_hi = high_res.size().width;   // number of cols
  int size_hi = height_hi*width_hi;       // Total number of pixels of input images
  // Pointer variables for both host and device computing
  UCHAR * F, *G, * R;

  // Memory Allocation for device computing (input and output images)
  cudaMalloc((void**)&F, size_hi*sizeof(UCHAR));
  cudaMalloc((void**)&G, size_low*sizeof(UCHAR));
  cudaMalloc((void**)&R, size_hi*sizeof(UCHAR));

  // Definition of Gaussian Spatial and Range Masks
  cv::Mat gaussianMask;
  int maskSum;
  float range_mask[256];

  // Definition of Sigmas
  const double sigma = sigma_spatial;
  const float sigmaRange = sigma_range;

  //  Creating spatial and range masks
  CreateGaussianMask(window_size, gaussianMask, maskSum, sigma);
  CreateGaussianrange(range_mask, sigmaRange);

  int spatialSize = gaussianMask.size().width * gaussianMask.size().height;
  int rangeSize = 256;

  UCHAR * Sm; float * Rm; // range and spatial mask

  // Allocating and copying data to device
  cudaMalloc((void**)&Sm, spatialSize*sizeof(UCHAR));
  cudaMalloc((void**)&Rm, rangeSize*sizeof(float));

  cudaMemcpy(G, low_res.data, size_low*sizeof(UCHAR), cudaMemcpyHostToDevice);           // result_imging image for device
  cudaMemcpy(F, high_res.data, size_hi*sizeof(UCHAR), cudaMemcpyHostToDevice);           // Original image to device
  cudaMemcpy(R, Result.data, size_hi*sizeof(UCHAR), cudaMemcpyHostToDevice);             // result_imging image for device
  cudaMemcpy(Sm, gaussianMask.data, spatialSize*sizeof(UCHAR), cudaMemcpyHostToDevice);  // spatial mask to device
  cudaMemcpy(Rm, range_mask, rangeSize*sizeof(float), cudaMemcpyHostToDevice);           // range mask to device

  int blocksNumber_x = std::ceil(width_hi/(float)THREADS_PER_BLOCK);
  int blocksNumber_y = std::ceil(height_hi/(float)THREADS_PER_BLOCK);

  const dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
  const dim3 blocksPerGrid(blocksNumber_x, blocksNumber_y);

  JBU_CUDA<<<blocksPerGrid, threadsPerBlock>>>(F, G, R, width_hi, height_hi, width_low, height_low, window_size,  Sm, Rm);
  
  cudaMemcpy(Result.data, R, size_hi*sizeof(UCHAR), cudaMemcpyDeviceToHost);

  // Free memory 
  cudaFree(F); cudaFree(G); cudaFree(R);
  cudaFree(Sm); cudaFree(Rm);
}


__host__
void upsample(cv::Mat& low_res, cv::Mat& high_res, cv::Mat& Result, int& window_size, double sigma_spatial, double sigma_range)
{

  cv::Mat D = low_res.clone();

  cv::Mat I = high_res.clone();

  int uf = std::log2(I.rows/D.rows);

  cv::Mat I_lo;

  for (size_t i = 0; i < uf; i++)
  {
      cv::resize(D, D, cv::Size(2 * D.size().width, 2 * D.size().height), cv::INTER_LINEAR);
      cv::resize(I, I_lo, D.size(), cv::INTER_LINEAR);
      /* Filtering */
      cv::Mat D_temp = D.clone();
      JB_Filtering(D, I_lo, D_temp, window_size, sigma_spatial, sigma_range);
      D = D_temp.clone();
  }

  cv::resize(D, D, I.size(), cv::INTER_LINEAR);
  /* filter */
  JB_Filtering(D, I, Result, window_size, sigma_spatial, sigma_range);

}

void Disparity2PointCloud(
  const std::string& output_file,
  int height, int width, cv::Mat& disparities,
  const int& window_size,
  const int& dmin, const double& baseline, const double& focal_length)
{
  std::stringstream out3d;
  out3d << output_file << ".xyz";
  std::ofstream outfile(out3d.str());
  #pragma omp parallel for
  for (int i = 0; i < height - window_size; ++i) {
    std::cout << "Reconstructing 3D point cloud from disparities... " << std::ceil(((i) / static_cast<double>(height - window_size + 1)) * 100) << "%\r" << std::flush;
    for (int j = 0; j < width - window_size; ++j) {
      if (disparities.at<uchar>(i, j) == 0) continue;

      int d = disparities.at<uchar>(i, j) + dmin;
      double u1 = j - (width/2), u2 = u1 + d, v1 = i - (height/2);

      // TODO
      const double Z = (baseline*focal_length)/d;
      const double X = (-baseline)*(u1+u2)/(2*d);
      const double Y = baseline*v1/d;
	  //
      outfile << X/1000.0 << " " << Y/1000.0 << " " << Z/1000.0 << std::endl;
    }
  }
  std::cout << "Reconstructing 3D point cloud from disparities... Done.\r" << std::flush;
  std::cout << std::endl;
}