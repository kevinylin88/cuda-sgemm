#include "matrix.h"
#include <algorithm>
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
int check_mat(matrix mat1, matrix mat2, matrix mat3){
    // Placeholder: implementation to be added 
    return 0;
}

void sgemm_naive(matrix mat1, matrix mat2, matrix mat3){
    if(check_mat(mat1, mat2, mat3) == 1) return;
    // IMPLEMENTATION TO BE ADDED,
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
