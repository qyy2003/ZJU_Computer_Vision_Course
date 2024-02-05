#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/opencv.hpp"
#include<stdio.h>
#include <string>
#include <cmath>
#include <filesystem>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

namespace fs = std::filesystem;
using recursive_directory_iterator = std::filesystem::recursive_directory_iterator;
using namespace cv;

Mat correct_with_eye(Mat img,int x1,int y1,int x2,int y2);