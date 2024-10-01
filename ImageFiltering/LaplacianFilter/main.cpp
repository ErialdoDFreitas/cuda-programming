//
// Implementação de Filtro Laplaciano usando CUDA
//
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <stdio.h>

using namespace std;


extern "C" bool laplacianFilter_GPU_wrapper(const cv::Mat& input, cv::Mat& output);
extern "C" bool laplacianFilter_CPU(const cv::Mat& input, cv::Mat& output);


// Program main
int main(int argc, char** argv) {

   // Nome da imagem
   string image_name = "../Images/Samples/sample1_in";

   // Definindo nomes para imagens de entrada & saída
   string input_file =  image_name+".jpg";
   string output_file_cpu = image_name+"_out_cpu.jpeg";
   string output_file_gpu = image_name+"_out_gpu.jpeg";

   // Lendo a imagem de entrada
   cv::Mat srcImage = cv::imread(input_file, cv::IMREAD_UNCHANGED);
   if(srcImage.empty())
   {
      std::cout<<"Image Not Found: "<< input_file << std::endl;
      return -1;
   }
   cout <<"\n Tamanho da Imagem de entrada: "<<srcImage.cols<<", "<<srcImage.rows<<", "<<srcImage.channels()<<"\n";

   // convertendo RGB para gray scale
   cv::cvtColor(srcImage, srcImage, cv::COLOR_BGR2GRAY);

   // Declarando a imagem de saída  
   cv::Mat dstImage(srcImage.size(), srcImage.type());

   // Executando filtro laplaciano na GPU ---------------------------------------------------------------
   laplacianFilter_GPU_wrapper(srcImage, dstImage);
   // normalizando para 0-255
   dstImage.convertTo(dstImage, CV_32F, 1.0 / 255, 0);
   dstImage*=255;
   // Gerando imagem de saída
   imwrite(output_file_gpu, dstImage);

   // Executando filtro laplaciano on CPU ---------------------------------------------------------------
   laplacianFilter_CPU(srcImage, dstImage);
   // normalizando para 0-255
   dstImage.convertTo(dstImage, CV_32F, 1.0 / 255, 0);
   dstImage*=255;
   // Gerando imagem de saída
   imwrite(output_file_cpu, dstImage);
      
   return 0;
}