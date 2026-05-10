#include "matrix.h"
#include <algorithm>
#include <iostream>
#include <cuda_runtime.h>
#include <cstddef>
#include <cublas_v2.h>
// TODO:
// 1. Remove __device__ from check_mat since it is called on the CPU side.

// 2. std::cerr is only valid if check_mat remains a CPU function.

// 3. Prefer using k < mat1.cols in the loop since k iterates over A's columns / B's rows.

__device__ int check_mat(const matrix mat1, const matrix mat2, const matrix mat3){
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
    
    size_t row = blockDim.y * blockIdx.y + threadIdx.y;
    size_t col = blockDim.x * blockIdx.x + threadIdx.x;

    if(row >= mat3.rows || col >= mat3.cols) return;
    float sum = 0;
    for(size_t k = 0; k < mat2.rows; k++){
        sum += mat1.data[row * mat1.cols + k] * mat2.data[k * mat2.cols + col];
    }
    *(mat3.data + row * mat3.cols + col) = sum;
}

void sgemm_naive(matrix mat1, matrix mat2, matrix mat3){
    if(check_mat(mat1, mat2, mat3) == 1) return;
    dim3 block(16, 16);
    dim3 grid(
    (mat3.cols + block.x - 1) / block.x,
    (mat3.rows + block.y - 1) / block.y
    );// why do we assign such grid and block?

    sgemm_naive_gpu<<<grid, block>>>(mat1, mat2, mat3);
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
        mat1.cols,
        mat1.rows,
        mat2.rows,
        &alpha,
        mat2.data, mat1.cols, // lda: leading dimension of matrix B in column-major interpretation
        mat1.data, mat2.cols,
        &beta,
        mat3.data, mat1.cols
    );

    cublasDestroy(handle);
}
