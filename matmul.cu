#include <iostream>
#include <cuda_runtime.h>

#define N 1024

__global__ void matmul(float *A, float *B, float *C, int n){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < n && col < n){
        float sum = 0.0f;
        for(int k = 0; k < n; k++){
            sum += A[row*n + k] * B[k*n + col];
        }
        C[row*n + col] = sum;
    }
}

int main(){
    int n = N;
    size_t size = n * n * sizeof(float);

    //CPUneic
    float *h_A = new float[n * n];
    float *h_B = new float[n * n];
    float *h_C = new float[n * n];

    //初始化
    for(int i = 0; i < n * n; i++){
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
        h_C[i] = 0.0f;
    }

    //GPU内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    //从CPU到GPU搬运内存
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    //启动计算kernel
    dim3 threads(16, 16);
    dim3 blocks(n / 16, n / 16);
    matmul<<<blocks, threads>>>(d_A, d_B, d_C, n);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    std::cout << "C[0][0] = " << h_C[0] << ", expected " << n << std::endl;

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}