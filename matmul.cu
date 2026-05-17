#include "matrix.h"
#include <algorithm>
#include <iostream>
#include <cuda_runtime.h>
#include <cstddef>
#include <cublas_v2.h>

int check_mat(const matrix mat1, const matrix mat2, const matrix mat3){
    // Matrix dimension mismatch
    if(mat1.cols != mat2.rows || mat1.rows != mat3.rows || mat2.cols != mat3.cols){
        std::cerr << "Matrix dimension mismatch" << std::endl;
        return 1;
    }
    // NULL pointer
    if(mat1.data == nullptr || mat2.data == nullptr || mat3.data == nullptr){
        std::cerr << "Null pointer error" << std::endl;
        return 1;
    }
    // Invalid matrix size
    if(mat1.rows == 0 || mat1.cols == 0 ||
       mat2.rows == 0 || mat2.cols == 0 ||
       mat3.rows == 0 || mat3.cols == 0
    ){
        std::cerr << "Invalid matrix size" << std::endl;
        return 1;
    }
    return 0;
}

__global__ void sgemm_naive_gpu(matrix mat1, matrix mat2, matrix mat3){
    
    size_t row = blockDim.x * blockIdx.x + threadIdx.x;
    size_t col = blockDim.y * blockIdx.y + threadIdx.y;

    if(row >= mat3.rows || col >= mat3.cols) return;
    float sum = 0;
    for(size_t k = 0; k < mat1.cols; k++){
        sum += mat1.data[row * mat1.cols + k] * mat2.data[k * mat2.cols + col];
    }
    *(mat3.data + row * mat3.cols + col) = sum;
}

// Launch naive SGEMM kernel.
// Assumes mat1.data, mat2.data, and mat3.data are already allocated on GPU device memory.
void sgemm_naive(matrix mat1, matrix mat2, matrix mat3){
    if(check_mat(mat1, mat2, mat3) == 1) return;
    dim3 block(16, 16);
    dim3 grid(
    (mat3.cols + block.x - 1) / block.x,
    (mat3.rows + block.y - 1) / block.y
    );// why do we assign such grid and block?

    sgemm_naive_gpu<<<grid, block>>>(mat1, mat2, mat3);
}

__global__ void sgemm_v2_coalesced_gpu(matrix mat1, matrix mat2, matrix mat3){
    
    size_t row = blockDim.y * blockIdx.y + threadIdx.y;
    size_t col = blockDim.x * blockIdx.x + threadIdx.x;

    if(row >= mat3.rows || col >= mat3.cols) return;
    float sum = 0;
    for(size_t k = 0; k < mat1.cols; k++){
        sum += mat1.data[row * mat1.cols + k] * mat2.data[k * mat2.cols + col];
    }
    *(mat3.data + row * mat3.cols + col) = sum;
}

void sgemm_v2_coalesced(matrix mat1, matrix mat2, matrix mat3){
    if(check_mat(mat1, mat2, mat3) == 1) return ;
    dim3 block(16, 16);
    dim3 grid(
        (mat3.cols + block.x - 1) / block.x,
        (mat3.rows + block.y - 1) / block.y
    );
    sgemm_v2_coalesced_gpu<<<grid, block>>>(mat1, mat2, mat3);
}

#define TILE_SIZE 32 // TILE

__global__ void sgemm_v3_smem_gpu(matrix mat1, matrix mat2, matrix mat3){
    size_t row = blockDim.y * blockIdx.y + threadIdx.y;
    size_t col = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float shmat1[TILE_SIZE][TILE_SIZE]; // stands for shared_matrix
    __shared__ float shmat2[TILE_SIZE][TILE_SIZE];

    int count = 0;
    float sum = 0.0f;
    for(; count < (mat1.cols + TILE_SIZE - 1) / TILE_SIZE; count++){
        // load shmat1
        if(TILE_SIZE * count + threadIdx.x < mat1.cols && row < mat1.rows){
            shmat1[threadIdx.y][threadIdx.x] = mat1.data[row * mat1.cols + count * TILE_SIZE + threadIdx.x];
        }
        else{
            shmat1[threadIdx.y][threadIdx.x] = 0.0;
        }
        // load shmat2
        if(TILE_SIZE * count + threadIdx.y < mat2.rows && col < mat2.cols){
            shmat2[threadIdx.y][threadIdx.x] = mat2.data[(threadIdx.y + TILE_SIZE * count) * mat2.cols + col];
        }
        else{
            shmat2[threadIdx.y][threadIdx.x] = 0.0;
        }
        // __syncthreads() requires all threads in the same block to reach the barrier;
        // if only some boundary threads return early, the remaining threads may hang.
        __syncthreads();
        
        for(size_t k = 0; k < TILE_SIZE; k++){
            sum += shmat1[threadIdx.y][k] * shmat2[k][threadIdx.x];
        }
        __syncthreads();
    }
    if(row < mat3.rows && col < mat3.cols){
        mat3.data[row * mat3.cols + col] = sum;
    }
}

void sgemm_v3_smem(matrix mat1, matrix mat2, matrix mat3){
    if(check_mat(mat1, mat2, mat3) == 1) return;
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(
        (mat3.cols + block.x - 1) / block.x,
        (mat3.rows + block.y - 1) / block.y
    );
    sgemm_v3_smem_gpu<<<grid, block>>>(mat1, mat2, mat3);
}

#define MID_BLOCK 128
#define SM_BLOCK 8

__global__ void sgemm_v4_thread_tile_gpu(matrix mat1, matrix mat2, matrix mat3){
    if(check_mat(mat1, mat2, mat3) == 1) return;
    size_t row = blockDim.x * blockIdx.x + threadIdx.x;
    size_t col = blockDim.y * blockIdx.y + threadIdx.y;

    __shared__ float shmat1[MID_BLOCK][MID_BLOCK];
    __shared__ float shmat2[MID_BLOCK][MID_BLOCK];

    int count = 0;
    for(; count < (mat1.cols + MID_BLOCK - 1) / MID_BLOCK; count++){
        // load shmat1
        if(TILE_SIZE * count + threadIdx.x < mat1.cols && row < mat1.rows){
            shmat1[threadIdx.y][threadIdx.x] = mat1.data[row * mat1.cols + count * TILE_SIZE + threadIdx.x];
        }
        else{
            shmat1[threadIdx.y][threadIdx.x] = 0.0;
        }
        // load shmat2
        if(TILE_SIZE * count + threadIdx.y < mat2.rows && col < mat2.cols){
            shmat2[threadIdx.y][threadIdx.x] = mat2.data[(threadIdx.y + TILE_SIZE * count) * mat2.cols + col];
        }
        else{
            shmat2[threadIdx.y][threadIdx.x] = 0.0;
        }
        __syncthreads();
        for(int k = 0; k < MID_BLOCK; k++){
            for(int i = 0; i < SM_BLOCK; i++){
                
                for(int j = 0; j < SM_BLOCK; j++){

                }
            }
        }
    }
}

void sgemm_cublas(matrix mat1, matrix mat2, matrix mat3){
    if(check_mat(mat1, mat2, mat3) == 1){return;}
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;

    // in cuBLAS: C = A * B is C^T = B^T × A^T
    cublasSgemm(
        handle,
        CUBLAS_OP_N, //no transpose
        CUBLAS_OP_N,
        mat2.cols,
        mat1.rows,
        mat1.cols,
        &alpha,
        mat2.data, mat2.cols, // lda: leading dimension of matrix B in column-major interpretation
        mat1.data, mat1.cols,
        &beta,
        mat3.data, mat3.cols
    );

    cublasDestroy(handle);
}
