#include "kernels.cuh"
#include <chrono>
using namespace std::chrono;


int main(int argc, const char **argv) 
{

  // Sample run
  // ./Advanced_Filtering 1 ../images/Lena/lena.png ../images/Lena/lena.png ../results/Lena/JB/ 5 51.8

  if (argc < 7)
  {
    std::cout << "Execution Format..." << std::endl;
    std::cout << "./Advanced_Filtering mode low_res/noisy_img guide_img output_img_dir window_size sigma_range" << std::endl;
    std::cout << "mode: (0) Bilateral Filter" << std::endl;
    std::cout << "      (1) Joint Bilateral Filter" << std::endl;
    std::cout << "      (2) Joint Bilateral Upsampling" << std::endl;
    std::cout << "      (3) Iterative Upsampling" << std::endl;
    return EXIT_FAILURE;
  }

  // Mode Selection
  int mode = std::stoi(argv[1]);

  // Load images 
  cv::Mat  low_res = cv::imread(argv[2], cv::IMREAD_GRAYSCALE); 
  cv::Mat  high_res = cv::imread(argv[3], cv::IMREAD_GRAYSCALE); 
  std::string output_dir = argv[4];
  std::string output_file = output_dir +"result.png";
  std::string time_file = output_dir + "time";
  cv::Mat result_img = cv::Mat::zeros(high_res.size(), cv::IMREAD_GRAYSCALE);  

  // Add noise to original image to see filtering results
  if (mode < 2)
  {
    cv::Mat noise(low_res.size(), low_res.type());
    uchar mean = 0;
    uchar stddev = 20;
    cv::randn(noise, mean, stddev);
    low_res += noise; // "input"

    std::string noisy_img_file = output_dir +"noisy.png";
    cv::imwrite(noisy_img_file, low_res);
  }

  // Safety Measurements
  if (high_res.size().height*high_res.size().width < low_res.size().height*low_res.size().width) {
    std::cerr << "Please Make Sure to Follow the Following Format" << std::endl;
    std::cerr << "The First Image Cannot be Larger in Dimensions than the Second Image" << std::endl;
    std::cout << "./gui_app mode low_res_img_path high_res_img_path" << std::endl;
    return EXIT_FAILURE;
  }
  if (!high_res.data) {
    std::cerr << "No high_res data" << std::endl;
    return EXIT_FAILURE;
  }
  if (!low_res.data) {
    std::cerr << "No low_res data" << std::endl;
    return EXIT_FAILURE;
  }

  /*Filtering Parameters*/
  int window_size = std::stoi(argv[5]);
  int half_win_size = window_size/2;
  const double sigma = half_win_size / 2.5; 
  const float sigmaRange = std::stof(argv[6]);

  std::cout << "high_res: " << high_res.size()     << std::endl;
  std::cout << "low_res: " << low_res.size()       << std::endl;
  std::cout << "result_img: " << result_img.size() << std::endl;
  std::cout << "\nSelected Parameters: "           << std::endl;
  std::cout << "window_size: " << window_size      << std::endl;
  std::cout << "half_win_size: " << half_win_size  << std::endl;
  std::cout << "Spatial Sigma: " << sigma          << std::endl;
  std::cout << "Range Sigma: " << sigmaRange       << std::endl;
  std::cout << "output file name: " << output_file << std::endl;
  std::cout << "output time name: " << time_file   << std::endl;

  std::stringstream out3d;
  std::ofstream outfile(out3d.str());

  switch (mode)
  {
  case 0:
  {
    std::cout << "Bilateral Filtering Mode Selected" << std::endl;

    auto start = high_resolution_clock::now();
    Bilateral_Filtering(low_res, result_img, window_size, sigma, sigmaRange);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "Disparity Computation Took: " << duration.count()/1000000.0 << " Seconds" << std::endl;
    std::stringstream out3d;
    out3d << time_file << ".txt";
    std::ofstream outfile(out3d.str());
    outfile << duration.count()/1000000.0;
    break;
  }
  case 1:
  {
    std::cout << "Joint Bilateral Filtering Mode Selected" << std::endl;

    auto start = high_resolution_clock::now();
    JB_Filtering(low_res, high_res, result_img, window_size, sigma, sigmaRange);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "Disparity Computation Took: " << duration.count()/1000000.0 << " Seconds" << std::endl;
    std::stringstream out3d;
    out3d << time_file << ".txt";
    std::ofstream outfile(out3d.str());
    outfile << duration.count()/1000000.0;
    break;
  }
  case 2:
  {
    std::cout << "Joint Bilateral Upsampling Mode Selected" << std::endl;

    auto start = high_resolution_clock::now();
    JBU_Filtering(low_res, high_res, result_img, window_size, sigma, sigmaRange);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "Disparity Computation Took: " << duration.count()/1000000.0 << " Seconds" << std::endl;
    std::stringstream out3d;
    out3d << time_file << ".txt";
    std::ofstream outfile(out3d.str());
    outfile << duration.count()/1000000.0;
    break;
  }
  case 3:
  {
    std::cout << "Iterative Upsampling Mode Selected" << std::endl;

    auto start = high_resolution_clock::now();
    upsample(low_res, high_res, result_img, window_size, sigma, sigmaRange);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "Disparity Computation Took: " << duration.count()/1000000.0 << " Seconds" << std::endl;
    std::stringstream out3d;
    out3d << time_file << ".txt";
    std::ofstream outfile(out3d.str());
    outfile << duration.count()/1000000.0;
    break;
  }
  default:
    break;
  }

  // Show guide img, noisy img and Resulting img
  // cv::namedWindow("F", cv::WINDOW_NORMAL);
  // cv::imshow("F", high_res);

  // cv::namedWindow("G", cv::WINDOW_NORMAL);
  // cv::imshow("G", low_res);

  // cv::namedWindow("R", cv::WINDOW_NORMAL);
  // cv::imshow("R", result_img);

  // cv::waitKey(0);

  cv::imwrite(output_file, result_img);

  return 0;
}