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

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

__global__ void sgemm_v4_register_gpu(matrix mat1, matrix mat2, matrix mat3){
    __shared__ float shmat1[BM][BK];
    __shared__ float shmat2[BK][BN];

    // 告诉每一个thread，处理8 * 8的行，blockDim.y * BM表示处理第几个block
    // threadIdx.y * TM表示这个thraed处理这个block的第几个8行
    int row_start = blockIdx.y * BM + threadIdx.y * TM;
    int col_start = blockIdx.x * BN + threadIdx.x * TN;
    float acc[TM][TN] = {0};

    int tid = blockDim.x * threadIdx.y + threadIdx.x;
    int batch = BM * BK / (blockDim.x * blockDim.y);

    int count = 0;
    // 每个count处理8行，每个thread处理4个元素
    for(; count < (mat1.cols - 1 + BK) / BK; count++){
        for(int i = 0; i < batch; i++){
            int remain_row = (tid * batch + i) / BK;
            int remain_col = (tid * batch + i) % BK;

            int global_row = blockIdx.y * BM + remain_row;
            int global_col = count * BK + remain_col;

            if(global_row < mat1.rows && global_col < mat1.cols){
                shmat1[remain_row][remain_col] =
                    mat1.data[global_row * mat1.cols + global_col];
            }
            else{
                shmat1[remain_row][remain_col] = 0.0f;
            }
        }
        for(int i = 0; i < batch; i++){
            int remain_row = (tid * batch + i) / BN;
            int remain_col = (tid * batch + i) % BN;

            int global_row = count * BK + remain_row;
            int global_col = blockIdx.x * BN + remain_col;

            if(global_row < mat2.rows && global_col < mat2.cols){
                shmat2[remain_row][remain_col] =
                    mat2.data[global_row * mat2.cols + global_col];
            }
            else{
                shmat2[remain_row][remain_col] = 0.0f;
            }
        }
        __syncthreads();
        float vec_a[TM] = {0};
        float vec_b[TN] = {0};
        for(int k = 0; k < BK; k++){
            for(int i = 0; i < TM; i++){
                vec_a[i] = shmat1[threadIdx.y * TM + i][k];
            }
            for(int j = 0; j < TN; j++){
                vec_b[j] = shmat2[k][threadIdx.x * TN + j];
            }
            for(int i = 0; i < TM; i++){
                for(int j = 0; j < TN; j++){
                    acc[i][j] += vec_a[i] * vec_b[j];
                }
            }
        }
        __syncthreads();
    }
    for(int i = 0; i < TM; i++){
            for(int j = 0; j < TN; j++){
                if(row_start + i < mat3.rows
                    && col_start + j < mat3.cols){
                    mat3.data[(row_start + i) * mat3.cols + col_start + j] = acc[i][j];
                }
            }
        }
}

void sgemm_v4_register(matrix mat1, matrix mat2, matrix mat3){
    if(check_mat(mat1, mat2, mat3) ==1) return;
    dim3 block(BN / TN, BM / TM);
    dim3 grid(
        (mat3.cols + BN - 1) / BN,
        (mat3.rows + BM - 1) / BM
    );
    sgemm_v4_register_gpu<<<grid, block>>>(mat1, mat2, mat3);
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
