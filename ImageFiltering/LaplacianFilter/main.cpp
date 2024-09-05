#include <iostream>
#include <stdio.h>
#include <string>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;

extern "C" bool laplacianFilter_GPU_wrapper(const cv::Mat& input, cv:Mat& output);

int main(int argc, char** argv ) {

    // nome da imagem
    string imageName = "../Images/Samples/sample1_in.jpg";
    cout << imageName << endl; 
}