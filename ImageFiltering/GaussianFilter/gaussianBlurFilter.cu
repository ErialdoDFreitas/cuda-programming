#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

#pragma once
#ifdef __INTELLISENSE__
    void __syncthreads();
#endif

#define MAX_KERNEL_WIDTH 441
__constant__ double K[MAX_KERNEL_WIDTH];

__global__ void Gaussian(double*, double*, int, int, int, int);
__host__ void GenerateGaussianKernel(double*, int, double);
__host__ void errorCatch(cudaError_t, string);
template<typename T> size_t vectorBytesSize(const typename vector<T>&);

// Wrapper para chamar o filtro Gaussiano na GPU
extern "C" bool gaussianFilter_GPU_wrapper(const cv::Mat& inputImg, cv::Mat& outputImg) {
    vector<double> hostIn, hostKernel, hostOut;
    double* deviceIn, *deviceOut;
    int inCols = inputImg.cols, inRows = inputImg.rows;
    int kernelDim = 5;
    kernelRadius = floor(kernelDim / 2.0);
    int outCols = inCols - (kDim - 1);
    int outRows = inRows - (kDim - 1);
    double blockWidth = 8;

    // Eventos CUDA para medir o tempo de execução
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Transfere os dados da imagem OpenCV para um vetor de doubles (imagem linearizada)
    hostIn.assign(inputImg.ptr<double>(0), inputImg.ptr<double>(0) + inputImg.total());
    hostOut.resize(inCols * inRows, 0);

    // Gera o kernel Gaussiano
    hostKernel.resize(pow(kernelDim, 2), 0);
    GenerateGaussianKernel(hostKernel, kernelDim, kernelRadius);

    // Aloca memória no device (GPU) para a imagem de entrada e saída
    errorCatch(cudaMalloc(void**)& deviceIn, vectorBytesSize(hostIn));
    errorCatch(cudaMemcpy(deviceIn, hostIn.data(), vectorBytesSize(hostIn), cudaMemcpyHostToDevice));
    errorCatch(cudaMalloc(void**)& deviceOut, vectorBytesSize(hostOut));
    errorCatch(cudaMemcpyToSymbol(K, hostKernel.data(), vectorBytesSize(hostKernel)));

    // Inicia o temporizador
    cudaEventRecord(start);

    // Define a configuração de execução do kernel
    int blockWidthHalo = blockWidth + (kernelDim - 1);
    dim3 dimBlock(blockWidthHalo, blockWidthHalo);
    dim3 dimGrid(ceil(inCols / blockWidth), ceil(inRows / blockWidth));
    Gaussian<<<dimGrid, dimBlock, blockWidthHalo * blockWidthHalo * sizeof(double)>>>
        (deviceIn, deviceOut, kernelDim, inCols, outCols, outRows);
    errorCatch(cudaDeviceSynchronize());
    errorCatch(cudaMemcpy(hostOut.data(), deviceOut, vectorBytesSize(hostOut), cudaMemcpyDeviceToHost));

    // // Normaliza os valores da imagem de saída para o intervalo [0, 255]
    // double maxValue = *max_element(hostOut.begin(), hostOut.end());
    // for (auto& value : hostOut)
    //     value = (value / maxValue) * 255;

    // // Converte a imagem de saída (vetor de doubles) para uma imagem OpenCV
    // vector<int> toInt(hostOut.begin(), hostOut.end());
    // Mat blurImg = Mat(toInt).reshape(0, outRows);
    
    // Converte a imagem de saída (vetor de doubles) para uma imagem OpenCV, sem normalização
    Mat blurImg = Mat(outRows, outCols, CV_64F, hostOut.data());
    blurImg.convertTo(outputImg, CV_8UC1);

    // Libera a memória alocada na GPU
    errorCatch(cudaFree(deviceIn));
    errorCatch(cudaFree(deviceOut));

    // Sincroniza os eventos para garantir que o tempo de execução esteja correto
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    
    // Calcula o tempo de execução em milissegundos
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "\n Tempo de processamento na GPU (ms): " << milliseconds << "\n";

    return true; // Indica sucesso.
}


// Definição do Kernel Cuda, para execução da convolução da imagem com o kernel
__global__
void Gaussian(double* In, double* Out, int kernelDim, int inWidth, int outWidth, int outHeight) {
    extern __shared__ double loadIn[]; // Mem. compart. para armazenar o bloco de entrada com céluas de halo
    int trueDimX = blockDim.x + (kernelDim - 1);
    int trueDimY = blockDim.y + (kernelDim - 1);
    int col = (blockIdx.x * trueDimX) + threadIdx.x;
    int row = (blockIdx.y * trueDimY) + threadIdx.y;

    // Verifica se a thread está dentro dos limites válidos da imagem de saída
    if (col < outWidth && row < outHeight) {
        loadIn[threadOdx.y * blockDim.x + threadIdx.x] = In[row * inWidth + col];
        __syncthreads();

        if (threadIdx.x < trueDimX && threadIdx.y < trueDimY) {
            double acc = 0;
            for (int i=0; i < kernelDim; i++) 
                for (int j=0; j < kernelDim; j++) 
                    acc += loadIn[(threadIdx.y + i) * blockDim.x + (threadIdx.x + j)] * K[(i * kernelDim) + j];
            Out[row * outWidth + col] = acc;
    
        }
    } else {
        loadIn[threadIdx.y * blockDim.x + threadIdx.x] = 0.0;
    }

}


// Função para gerar o kernel Gaussiano, baseado na equação da distribuição Gaussiana
__host__
void GenerateGaussianKernel(vector<double>& k, int dim, int radius) {
    double stDev = 1.0; // Define o desvio padrão da distribuição Gaussiana
    double pi = 355.0 / 113.0;
    double constantEq = 1.0 / (2.0 * pi * pow(stDev, 2)); // Constante da equação da distribuição Gaussiana

    for (int i = -radius; i < radius + 1; i++) {
        for (int j = -radius; j < radius + 1; j++) {
            K[(i + radius) * dim + (j + radius)]
                = constantEq * (1 / exp((pow(i, 2) + pow(j, 2)) / (2 * pow(stDev, 2))));
        }
    }
}


// Função para capturar e tratar erros de chamadas CUDA
__host__
void errorCatch(cudaError_t err) { //, string msg) {
    if (err != cudaSuccess) {
        cerr << "CUDA error: " << cudaGetErrorString(err) << endl;
        // cerr << msg << " (error code " << err << ")\n";
        exit(EXIT_FAILURE);
    }
}


// Função para calcular o tamanho em bytes de um vetor (usado para alocar memória na GPU)
template<typename T>
size_t vectorBytesSize(const typename vector<T>& vec) {
    return vec.size() * sizeof(T);
}