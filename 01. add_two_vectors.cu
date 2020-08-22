// std C libs
#include <stdio.h>
#include <stdlib.h>

// CUDA libs
#include <cuda_runtime_api.h>

// vector size
#define N 8 

/// <summary>Adding vectors on GPU on blocks</summary>
/// <param name="vec1">- first vector</param>
/// <param name="vec2">- second vector</param>
/// <param name="vec3">- third vector with sum</param>
/// <param name="n">- vectors size</param>
__global__ 
void add_vec_GPU_v1(int* vec1, int* vec2, int* vec3, size_t n);


/// <summary>Adding vectors on GPU on threads</summary>
/// <param name="vec1">- first vector</param>
/// <param name="vec2">- second vector</param>
/// <param name="vec3">- third vector with sum</param>
/// <param name="n">- vectors size</param>
__global__ 
void add_vec_GPU_v2(int* vec1, int* vec2, int* vec3, size_t n);

/// <summary>Adding vectors on GPU on blocks and threads</summary>
/// <param name="vec1">- first vector</param>
/// <param name="vec2">- second vector</param>
/// <param name="vec3">- third vector with sum</param>
/// <param name="n">- vectors size</param>
__global__
void add_vec_GPU_v3(int* vec1, int* vec2, int* vec3, size_t n);

/// <summary>Adding vectors on CPU in simple loop</summary>
/// <param name="vec1">- first vector</param>
/// <param name="vec2">- second vector</param>
/// <param name="vec3">- third vector with sum</param>
/// <param name="n">- vectors size</param>
void add_vec_CPU(int* vec1, int* vec2, int* vec3, size_t n);

/// <summary>Function to print vector</summary>
/// <param name="label">- vector label</param>
/// <param name="vec">- vector to print</param>
/// <param name="n">- vector size</param>
void print_vec(const char* label, int* vec, size_t n);

/// <summary>Checks if the CUDA function has been carried out correctly.
/// If not, it terminates the program with an error status.</summary>
/// <param name="status">- function execution status (correct is the same as <c>cudaSucces</c>)</param>
void catchException(cudaError_t status);

int main()
{
    cudaError_t status = cudaSuccess;

    int vec1[N] = { 3, 4, 8, 3, 98, -12, 743, 8 };
    int vec2[N] = { 7, -2, 3, 823, -12, 843, 912, 0 };

    int vec3_CPU[N], vec3_GPU_v1[N], vec3_GPU_v2[N], vec3_GPU_v3[N];
    int* dev_vec1, *dev_vec2, *dev_vec3_v1, *dev_vec3_v2, *dev_vec3_v3;
    
    // computing on CPU
    add_vec_CPU(vec1, vec2, vec3_CPU, N);

    // memory allocation on GPU
    status = cudaMalloc((void**)&dev_vec1, N * sizeof(dev_vec1[0]));
    catchException(status);
    status = cudaMalloc((void**)&dev_vec2, N * sizeof(dev_vec2[0]));
    catchException(status);
    status = cudaMalloc((void**)&dev_vec3_v1, N * sizeof(dev_vec3_v1[0]));
    catchException(status);
    status = cudaMalloc((void**)&dev_vec3_v2, N * sizeof(dev_vec3_v2[0]));
    catchException(status);
    status = cudaMalloc((void**)&dev_vec3_v3, N * sizeof(dev_vec3_v3[0]));
    catchException(status);
    print_vec("vector 1:\t\t", vec1, N);
    print_vec("vector 2:\t\t", vec2, N);
    printf("==========\n");

    // copy to GPU
    status = cudaMemcpy(dev_vec1, vec1, N * sizeof(vec1[0]), cudaMemcpyHostToDevice);
    catchException(status);
    status = cudaMemcpy(dev_vec2, vec2, N * sizeof(vec2[0]), cudaMemcpyHostToDevice);
    catchException(status);

    // computing on GPU v1
    add_vec_GPU_v1 << < N, 1 >> > (dev_vec1, dev_vec2, dev_vec3_v1, N);

    // computing on GPU v2
    add_vec_GPU_v2 << < 1, N >> > (dev_vec1, dev_vec2, dev_vec3_v2, N);

    // computing on GPU v3
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    add_vec_GPU_v3 << < blocksPerGrid, threadsPerBlock >> > (dev_vec1, dev_vec2, dev_vec3_v3, N);

    // copy result to CPU
    status = cudaMemcpy(vec3_GPU_v1, dev_vec3_v1, N * sizeof(dev_vec3_v1[0]), cudaMemcpyDeviceToHost);
    catchException(status);
    status = cudaMemcpy(vec3_GPU_v2, dev_vec3_v2, N * sizeof(dev_vec3_v2[0]), cudaMemcpyDeviceToHost);
    catchException(status);
    status = cudaMemcpy(vec3_GPU_v3, dev_vec3_v3, N * sizeof(dev_vec3_v3[0]), cudaMemcpyDeviceToHost);
    catchException(status);

    // print results
    print_vec("vector 3 on CPU:\t", vec3_CPU, N);
    print_vec("vector 3 on GPU v1:\t", vec3_GPU_v1, N);
    print_vec("vector 3 on GPU v2:\t", vec3_GPU_v2, N);
    print_vec("vector 3 on GPU v3:\t", vec3_GPU_v3, N);

    // free memory on GPU
    cudaFree(dev_vec1);
    cudaFree(dev_vec2);
    cudaFree(dev_vec3_v1);
    cudaFree(dev_vec3_v2);
    cudaFree(dev_vec3_v3);

    exit(EXIT_SUCCESS);
}

__global__
void add_vec_GPU_v1(int* vec1, int* vec2, int* vec3, size_t n) {
    int tid = blockIdx.x;
    if (tid < n) {
        vec3[tid] = vec1[tid] + vec2[tid];
    }
}

__global__
void add_vec_GPU_v2(int* vec1, int* vec2, int* vec3, size_t n) {
    int tid = threadIdx.x;
    if (tid < n) {
        vec3[tid] = vec1[tid] + vec2[tid];
    }
}

__global__
void add_vec_GPU_v3(int* vec1, int* vec2, int* vec3, size_t n) {
    
    // thread ID for each thread calculating corresponding vector index
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < n) {
        vec3[tid] = vec1[tid] + vec2[tid];
        tid += gridDim.x * blockDim.x;
    }
}

void add_vec_CPU(int* vec1, int* vec2, int* vec3, size_t n) {
    int i;
    for (i = 0; i < n; i++)
        vec3[i] = vec1[i] + vec2[i];
}

void print_vec(const char* label, int* vec, size_t n) {
    int i;
    printf("%s [", label);
    for (i = 0; i < n - 1; i++)
        printf("%4d, ", vec[i]);
    printf("%4d]\n", vec[n - 1]);
}

void catchException(cudaError_t status) {
    if (status != cudaSuccess) {
        printf("%s\n", cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}