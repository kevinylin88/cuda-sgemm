#include <iostream>
#include <cuda_runtime.h>
#include <cstring>
#include "matrix.h"
#include <random>
#include <algorithm>
using namespace std;
#define MEASURE_TIME 20

std::random_device rd;
std::mt19937 gen(rd());
float rand_float(float l, float r){
    std::uniform_real_distribution<float> dist(l, r);
    return dist(gen);
}

void test_matmul(matrix mat1, matrix mat2, matrix mat3, const std::string func_type){
    float result[MEASURE_TIME];
    float ms = 0;

    if(func_type == "cublas"){//cuBLAS branch
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        for(int i = 0; i < MEASURE_TIME; i++){
            //set mat3 to zero before each multiplication
            cudaMemset(mat3.data, 0, mat3.cols * mat3.rows * sizeof(float));
            cudaEventRecord(start);
            sgemm_cublas(mat1, mat2, mat3);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&ms, start, stop);
            result[i] = ms;
        }
        //Calculate the maximum, minimum, and median values.
        sort(result, result + MEASURE_TIME);
        float smaller = result[MEASURE_TIME / 2 - 1];
        float bigger = result[MEASURE_TIME / 2];
        float median = (smaller + bigger) / 2.0;
        float worst = result[MEASURE_TIME - 1];
        float best = result[0];
        cout << "matrix size: " << mat1.rows << " x " << mat1.cols << std::endl;
        cout << "cublas: median:" << median << " ms, best: " << best << " ms, worst: " << worst << " ms" << endl;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    else if(func_type == "mine"){//my branch
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        for(int i = 0; i < MEASURE_TIME; i++){
            //set mat3 to zero before each multiplication
            cudaMemset(mat3.data, 0, mat3.cols * mat3.rows * sizeof(float));
            cudaEventRecord(start);
            sgemm_v3_smem(mat1, mat2, mat3);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&ms, start, stop);
            result[i] = ms;
        }
        //Calculate the maximum, minimum, and median values.
        sort(result, result + MEASURE_TIME);
        float smaller = result[MEASURE_TIME / 2 - 1];
        float bigger = result[MEASURE_TIME / 2];
        float median = (smaller + bigger) / 2.0;
        float worst = result[MEASURE_TIME - 1];
        float best = result[0];
        cout << "matrix size: " << mat1.rows << " x " << mat1.cols << std::endl;
        cout << "mine: median:" << median << " ms, best: " << best << " ms, worst: " << worst << " ms" << endl;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
}

int main(int argc, char** argv) {
    matrix h_mat1, h_mat2, h_mat3;
    matrix d_mat1, d_mat2, d_mat3;
    size_t test_set[] = {64, 128, 256, 512, 8000};
    int len = sizeof(test_set) / sizeof(size_t);

    for(int i = 0; i < len; i++){
        int n = test_set[i];

        //allocate memory on host and device
        h_mat1.cols = n, h_mat1.rows = n, h_mat2.cols = n, h_mat2.rows = n, h_mat3.cols = n, h_mat3.rows = n;
        d_mat1.cols = n, d_mat1.rows = n, d_mat2.cols = n, d_mat2.rows = n, d_mat3.cols = n, d_mat3.rows = n;
        h_mat1.data = new float[n * n];
        h_mat2.data = new float[n * n];
        h_mat3.data = new float[n * n];
        cudaMalloc(&d_mat1.data, n * n * sizeof(float));
        cudaMalloc(&d_mat2.data, n * n * sizeof(float));
        cudaMalloc(&d_mat3.data, n * n * sizeof(float));

        //assign random value to the matrix
        for(int j = 0; j < n * n; j++){
            h_mat1.data[j] = rand_float(0, 1);
            h_mat2.data[j] = rand_float(0, 1);
        }

        //transfer memory to device
        cudaMemcpy(d_mat1.data, h_mat1.data, n * n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_mat2.data, h_mat2.data, n * n * sizeof(float), cudaMemcpyHostToDevice);
        memset(h_mat3.data, 0, n * n * sizeof(float));
        cudaMemcpy(d_mat3.data, h_mat3.data, n * n * sizeof(float), cudaMemcpyHostToDevice);

        if(argc > 1 && std::string(argv[1]) == "cublas"){
            test_matmul(d_mat1, d_mat2, d_mat3, "cublas");
        }
        else if(argc != 1){
            cout << "invalid argument, only support argument 'cublas'" << endl;
            return 0;
        }
        else{
            test_matmul(d_mat1, d_mat2, d_mat3, "mine");
        }

        cudaFree(d_mat1.data);
        cudaFree(d_mat2.data);
        cudaFree(d_mat3.data);
        delete[] h_mat1.data;
        delete[] h_mat2.data;
        delete[] h_mat3.data;
    }
    return 0;
}