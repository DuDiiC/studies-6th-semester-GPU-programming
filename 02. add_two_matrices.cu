// std C libs
#include <stdio.h>
#include <stdlib.h>

// CUDA libs
#include <cuda_runtime_api.h>

// matrices size (N rows, M columns)
#define N 3
#define M 4

/// <summary>Adding vectors on GPU on N threads</summary>
/// <param name="matrix1">- first matrix</param>
/// <param name="matrix2">- second matrix</param>
/// <param name="matrix3">- third matrix with sum</param>
/// <param name="n">- matrix size</param>
/// <param name="m">- matrix size</param>
__global__ 
void add_matrix_GPU(int* matrix1, int* matrix2, int* matrix3, size_t n, size_t m);

/// <summary>Adding vectors on CPU in simple loop</summary>
/// <param name="matrix1">- first matrix</param>
/// <param name="matrix2">- second matrix</param>
/// <param name="matrix3">- third matrix with sum</param>
/// <param name="n">- matrix size</param>
/// <param name="m">- matrix size</param>
void add_matrix_CPU(int* matrix1, int* matrix2, int* matrix3, size_t n, size_t m);

/// <summary>Function to print matrix. NOTE: works well with numbers in [-99; 999] (3 chars include sign - minus or none)</summary>
/// <param name="label">- vector label</param>
/// <param name="matrix">- vector to print</param>
/// <param name="n">- vector size</param>
void print_matrix(const char* label, int* matrix, size_t n, size_t m);

/// <summary>Checks if the CUDA function has been carried out correctly.
/// If not, it terminates the program with an error status.</summary>
/// <param name="status">- function execution status (correct is the same as <c>cudaSucces</c>)</param>
void catchException(cudaError_t status);

int main() {
    cudaError_t status = cudaSuccess;

    int matrix1[N * M] = { 1, 2, 3, 2, 3, 4, 3, 4, 5, 13, 2, 2 };
    int matrix2[N * M] = { -2, -3, -4, -1, -2, -3, 5, 6, 7, -1, 12, 3 };

    int matrix3_GPU[N * M], matrix3_CPU[N * M];
    int *dev_matrix1, *dev_matrix2, *dev_matrix3;

    print_matrix("matrix 1:", matrix1, N, M);
    print_matrix("matrix 2:", matrix2, N, M);
    printf("==========\n");

    // computing on CPU
    add_matrix_CPU(matrix1, matrix2, matrix3_CPU, N, M);
    print_matrix("matrix 3 on CPU:", matrix3_CPU, N, M);

    // memory allocation on GPU
    status = cudaMalloc((void**)&dev_matrix1, N * M * sizeof(dev_matrix1[0]));
    catchException(status);
    status = cudaMalloc((void**)&dev_matrix2, N * M * sizeof(dev_matrix2[0]));
    catchException(status);
    status = cudaMalloc((void**)&dev_matrix3, N * M * sizeof(dev_matrix3[0]));
    catchException(status);

    // copy to GPU
    status = cudaMemcpy(dev_matrix1, matrix1, N * M * sizeof(matrix1[0]), cudaMemcpyHostToDevice);
    catchException(status);
    status = cudaMemcpy(dev_matrix2, matrix2, N * M * sizeof(matrix2[0]), cudaMemcpyHostToDevice);
    catchException(status);

    // computing on GPU
    int numBlocks = 1;
    dim3 threadsPerBlock(N, M);
    add_matrix_GPU << < numBlocks, threadsPerBlock >> > (dev_matrix1, dev_matrix2, dev_matrix3, N, M);
    
    // copy result on CPU
    status = cudaMemcpy(matrix3_GPU, dev_matrix3, N * M * sizeof(dev_matrix3[0]), cudaMemcpyDeviceToHost);
    catchException(status);

    print_matrix("matrix 3 on GPU:", matrix3_GPU, N, M);

    // free memory on GPU
    cudaFree(dev_matrix1);
    cudaFree(dev_matrix2);
    cudaFree(dev_matrix3);

    exit(EXIT_SUCCESS);
}

__global__
void add_matrix_GPU(int* matrix1, int* matrix2, int* matrix3, size_t n, size_t m) {
    int i = threadIdx.x;
    int j = threadIdx.y;
    if (i < n) {
        matrix3[i * m + j] = matrix1[i * m + j] + matrix2[i * m + j];
    }
}

void add_matrix_CPU(int* matrix1, int* matrix2, int* matrix3, size_t n, size_t m) {
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            matrix3[i * m + j] = matrix1[i * m + j] + matrix2[i * m + j];
        }
    }
}

void print_matrix(const char* label, int* matrix, size_t n, size_t m) {
    int i, j;
    printf("%s\n", label);
    for (i = 0; i < n; i++)  {
        printf("[");
        for (j = 0; j < m; j++) {
            printf("%4d", matrix[i * m + j]);
        }
        printf("]\n");
    }
}

void catchException(cudaError_t status) {
    if (status != cudaSuccess) {
        printf("%s\n", cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}