#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/opencv.hpp"
#include<stdio.h>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;
using recursive_directory_iterator = std::filesystem::recursive_directory_iterator;
using namespace cv;

int mark_eye(std::string input_path = "att-face/s0",std::string output_path = "ATT-eye-location/s0/");