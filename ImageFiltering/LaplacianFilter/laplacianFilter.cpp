//
// C++ implementation of Laplacian Filter
//
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <stdio.h>

using namespace std;
using namespace cv;
struct timeval t1, t2;

// O wrapper usado para chamar a função laplacianFilter na CPU
extern "C" void laplacianFilter_CPU(const cv::Mat& input, cv::Mat& output)
{
   cv::Mat input_gray;
   int kernel_size = 3;
   int scale = 1;
   int delta = 0;
 
   int64 t0 = cv::getTickCount();

   /// Removendo ruído borrando com um Filtro Gaussiano
   GaussianBlur(input, input, Size(3,3), 0, 0, BORDER_DEFAULT);

   // Aplicação do filtro Laplaciano
   Laplacian(input, output, CV_16S, kernel_size, scale, delta, BORDER_DEFAULT);

   int64 t1 = cv::getTickCount();
   double secs = (t1-t0)/cv::getTickFrequency();

   cout<< "\n Tempo de processamento na CPU (ms): " << secs*1000 << "\n";   
}