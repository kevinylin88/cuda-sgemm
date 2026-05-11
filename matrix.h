#ifndef matrix_h
#define matrix_h
#include <stdio.h>
#include <cstddef>
typedef struct matrix
{
    size_t rows;
    size_t cols;
    float* data;
}matrix;

void sgemm_cublas(matrix mat1, matrix mat2, matrix mat3);
void sgemm_naive(matrix mat1, matrix mat2, matrix mat3);
void sgemm_v2_coalesced(matrix mat1, matrix mat2, matrix mat3);
void sgemm_v3_smem(matrix mat1, matrix mat2, matrix mat3);
#endif