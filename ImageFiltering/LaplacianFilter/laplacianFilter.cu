//
// CUDA implementation of Laplacian Filter
//
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <stdio.h>
#include <cuda.h>
#include "cuda_runtime.h"

#define BLOCK_SIZE      16
#define FILTER_WIDTH    3       
#define FILTER_HEIGHT   3       

using namespace std;

// Rodando a aplicação do filtro Laplaciano usando a GPU
__global__ void laplacianFilter(unsigned char *srcImage, unsigned char *dstImage, unsigned int width, unsigned int height)
{
   // Cálculo da posição x e y da thread na imagem
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;

   // Máscara do filtro Laplaciano
   float kernel[3][3] = {0, -1, 0, -1, 4, -1, 0, -1, 0};
   //float kernel[3][3] = {-1, -1, -1, -1, 8, -1, -1, -1, -1};   

   // Verifica se a thread está dentro dos limites válidos da imagem
   if((x >= FILTER_WIDTH / 2) && (x < (width - FILTER_WIDTH / 2)) && (y >= FILTER_HEIGHT / 2) && (y < (height - FILTER_HEIGHT / 2)))
   {
       float sum = 0;

       // Aplica a convolução do kernel Laplaciano na imagem
       for(int ky = -FILTER_HEIGHT / 2; ky <= FILTER_HEIGHT / 2; ky++) {
           for(int kx = -FILTER_WIDTH / 2; kx <= FILTER_WIDTH / 2; kx++) {
               // Acessa os pixels vizinhos usando a máscara e multiplica pelos valores da máscara (kernel)
               float fl = srcImage[((y + ky) * width + (x + kx))];
               sum += fl * kernel[ky + FILTER_HEIGHT / 2][kx + FILTER_WIDTH / 2];
           }
       }
       // Escreve o valor resultante no pixel de saída
       dstImage[(y * width + x)] = sum;
   }
}


// O wrapper para chamar a função laplacianFilter na GPU
extern "C" void laplacianFilter_GPU_wrapper(const cv::Mat& input, cv::Mat& output)
{
    // Eventos CUDA para medir o tempo de execução
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Calcula o tamanho da imagem em bytes
    const int inputSize = input.cols * input.rows;
    const int outputSize = output.cols * output.rows;

    // Ponteiros para memória na GPU
    unsigned char *d_input, *d_output;

    // Aloca memória na GPU para as imagens de entrada e saída
    cudaMalloc<unsigned char>(&d_input, inputSize);
    cudaMalloc<unsigned char>(&d_output, outputSize);

    // Copia a imagem de entrada do host (CPU) para a GPU (device)
    cudaMemcpy(d_input, input.ptr(), inputSize, cudaMemcpyHostToDevice);

    // Define o tamanho do bloco de threads
    const dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    // Define o tamanho da grade de blocos
    const dim3 grid((output.cols + block.x - 1) / block.x, (output.rows + block.y - 1) / block.y);

    // Inicia o temporizador
    cudaEventRecord(start);

    // Chama o kernel CUDA para aplicar o filtro Laplaciano na imagem
    laplacianFilter<<<grid, block>>>(d_input, d_output, output.cols, output.rows);

    // Para o temporizador
    cudaEventRecord(stop);

    // Copia a imagem processada da GPU de volta para a CPU
    cudaMemcpy(output.ptr(), d_output, outputSize, cudaMemcpyDeviceToHost);

    // Libera a memória alocada na GPU
    cudaFree(d_input);
    cudaFree(d_output);

    // Sincroniza os eventos para garantir que o tempo de execução esteja correto
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    
    // Calcula o tempo de execução em milissegundos
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "\n Tempo de processamento na GPU (ms): " << milliseconds << "\n";
}
