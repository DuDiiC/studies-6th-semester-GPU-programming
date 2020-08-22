// KOMENTARZ:
//
// Testowano na:
//      - CPU: Intel Core i7-7700HQ (2.8 - 3.8 GHz, 6MB cache)
//      - GPU: NVIDIA GeFore RTX 1050 (2048 MB GDDR5) 
//
// W przypadku wywolywania metod obliczania sumy wektorow na roznych parametrach jadra,
// mozemy zauwazyc zdecydowana roznice w czasie obliczen. Mianowicie, im wiecej watkow
// w pojedynczym bloku wykorzystamy, tym obliczenia wykonuja sie szybciej. 
//
// ===================================================================


// std C libs
#include <stdio.h>
#include <stdlib.h>

// CUDA libs
#include <cuda_runtime_api.h>

// vector size
#define N 10000

/// <summary>Adding vectors on GPU on blocks and threads</summary>
/// <param name="vec1">- first vector</param>
/// <param name="vec2">- second vector</param>
/// <param name="vec3">- third vector with sum</param>
/// <param name="n">- vectors size</param>
__global__
void add_vec_GPU(int* vec1, int* vec2, int* vec3, size_t n);

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

/// <summary></summary>
/// <param name="vec1"></param>
/// <param name="vec2"></param>
/// <param name="n"></param>
/// <returns></returns>
int equal_vectors(int* vec1, int* vec2, size_t n);

/// <summary>Checks if the CUDA function has been carried out correctly.
/// If not, it terminates the program with an error CUDA_status.</summary>
/// <param name="CUDA_status">- function execution CUDA_status (correct is the same as <c>cudaSucces</c>)</param>
void catchException(cudaError_t status);

int main() {

    cudaError_t CUDA_status = cudaSuccess;
    int i;

    int vec1[N];
    int vec2[N];

    // set start values
    for (i = 0; i < N; i++) {
        vec1[i] = i * i;
        vec2[i] = -i;
    }

    int vec3_CPU[N],
        vec3_GPU_v1[N],
        vec3_GPU_v2[N], 
        vec3_GPU_v3[N];
    
    int *dev_vec1, 
        *dev_vec2, 
        *dev_vec3_v1, 
        *dev_vec3_v2, 
        *dev_vec3_v3;

    float   calc_t_GPU_v1,
            calc_t_GPU_v2,
            calc_t_GPU_v3;

    cudaEvent_t start_GPU_v1,
                start_GPU_v2,
                start_GPU_v3,
                stop_GPU_v1,
                stop_GPU_v2,
                stop_GPU_v3;

    CUDA_status = cudaEventCreate(&start_GPU_v1); catchException(CUDA_status);
    CUDA_status = cudaEventCreate(&start_GPU_v2); catchException(CUDA_status);
    CUDA_status = cudaEventCreate(&start_GPU_v3); catchException(CUDA_status);
    CUDA_status = cudaEventCreate(&stop_GPU_v1); catchException(CUDA_status);
    CUDA_status = cudaEventCreate(&stop_GPU_v2); catchException(CUDA_status);    
    CUDA_status = cudaEventCreate(&stop_GPU_v3); catchException(CUDA_status);

    // computing on CPU
    add_vec_CPU(vec1, vec2, vec3_CPU, N);

    // memory allocation on GPU
    CUDA_status = cudaMalloc((void**)&dev_vec1, N * sizeof(dev_vec1[0]));
    catchException(CUDA_status);
    CUDA_status = cudaMalloc((void**)&dev_vec2, N * sizeof(dev_vec2[0]));
    catchException(CUDA_status);
    CUDA_status = cudaMalloc((void**)&dev_vec3_v1, N * sizeof(dev_vec3_v1[0]));
    catchException(CUDA_status);
    CUDA_status = cudaMalloc((void**)&dev_vec3_v2, N * sizeof(dev_vec3_v2[0]));
    catchException(CUDA_status);
    CUDA_status = cudaMalloc((void**)&dev_vec3_v3, N * sizeof(dev_vec3_v3[0]));
    catchException(CUDA_status);

    // copy to GPU
    CUDA_status = cudaMemcpy(dev_vec1, vec1, N * sizeof(vec1[0]), cudaMemcpyHostToDevice);
    catchException(CUDA_status);
    CUDA_status = cudaMemcpy(dev_vec2, vec2, N * sizeof(vec2[0]), cudaMemcpyHostToDevice);
    catchException(CUDA_status);

    // COMPUTING ON GPU
    // ====================================================
    int threadsPerBlock_v1 = 32;
    int threadsPerBlock_v2 = 256;
    int threadsPerBlock_v3 = 2048;
    int blocksPerGrid_v1 = (N + threadsPerBlock_v1 - 1) / threadsPerBlock_v1;
    int blocksPerGrid_v2 = (N + threadsPerBlock_v2 - 1) / threadsPerBlock_v2;
    int blocksPerGrid_v3 = (N + threadsPerBlock_v3 - 1) / threadsPerBlock_v3;
    
    // computing on GPU v1
    // ====================================================
    
    // start of execution time measurement
    CUDA_status = cudaEventRecord(start_GPU_v1, 0);
    catchException(CUDA_status);
    
    add_vec_GPU << < blocksPerGrid_v1, threadsPerBlock_v1 >> > (dev_vec1, dev_vec2, dev_vec3_v1, N);
    
    // end of execution time measurement
    CUDA_status = cudaEventRecord(st op_GPU_v1);
    catchException(CUDA_status);
    CUDA_status = cudaEventSynchronize(stop_GPU_v1);
    catchException(CUDA_status);
    
    // calculate execution time
    CUDA_status = cudaEventElapsedTime(&calc_t_GPU_v1, start_GPU_v1, stop_GPU_v1);
    catchException(CUDA_status);

    // computing on GPU v2
    // ====================================================

    // start of execution time measurement
    CUDA_status = cudaEventRecord(start_GPU_v2, 0);
    catchException(CUDA_status);

    add_vec_GPU << < blocksPerGrid_v2, threadsPerBlock_v2 >> > (dev_vec1, dev_vec2, dev_vec3_v2, N);

    // end of execution time measurement
    CUDA_status = cudaEventRecord(stop_GPU_v2);
    catchException(CUDA_status);
    CUDA_status = cudaEventSynchronize(stop_GPU_v2);
    catchException(CUDA_status);

    // calculate execution time
    CUDA_status = cudaEventElapsedTime(&calc_t_GPU_v2, start_GPU_v2, stop_GPU_v2);
    catchException(CUDA_status);

    // computing on GPU v3
    // ====================================================

    // start of execution time measurement
    CUDA_status = cudaEventRecord(start_GPU_v3, 0);
    catchException(CUDA_status);

    add_vec_GPU << < blocksPerGrid_v3, threadsPerBlock_v3 >> > (dev_vec1, dev_vec2, dev_vec3_v3, N);

    // end of execution time measurement
    CUDA_status = cudaEventRecord(stop_GPU_v3);
    catchException(CUDA_status);
    CUDA_status = cudaEventSynchronize(stop_GPU_v3);
    catchException(CUDA_status);

    // calculate execution time
    CUDA_status = cudaEventElapsedTime(&calc_t_GPU_v3, start_GPU_v3, stop_GPU_v3);
    catchException(CUDA_status);

    // copy result to CPU
    CUDA_status = cudaMemcpy(vec3_GPU_v1, dev_vec3_v1, N * sizeof(dev_vec3_v1[0]), cudaMemcpyDeviceToHost);
    catchException(CUDA_status);
    CUDA_status = cudaMemcpy(vec3_GPU_v2, dev_vec3_v2, N * sizeof(dev_vec3_v2[0]), cudaMemcpyDeviceToHost);
    catchException(CUDA_status);
    CUDA_status = cudaMemcpy(vec3_GPU_v3, dev_vec3_v3, N * sizeof(dev_vec3_v3[0]), cudaMemcpyDeviceToHost);
    catchException(CUDA_status);

    // print results
    //print_vec("vector 1:\t\t", vec1, N);
    //print_vec("vector 2:\t\t", vec2, N);
    //printf("==========\n");
    //print_vec("vector 3 on CPU:\t", vec3_CPU, N);
    //print_vec("vector 3 on GPU v1:\t", vec3_GPU_v1, N);
    if (equal_vectors(vec3_CPU, vec3_GPU_v1, N)) printf("Obliczenia GPU v1 poprawne.\n");
    printf("Czas wykonania GPU v1:\t %f ms\n", calc_t_GPU_v1);
    //print_vec("vector 3 on GPU v2:\t", vec3_GPU_v2, N);
    if (equal_vectors(vec3_CPU, vec3_GPU_v2, N)) printf("Obliczenia GPU v2 poprawne.\n");
    printf("Czas wykonania GPU v2:\t %f ms\n", calc_t_GPU_v2);
    //print_vec("vector 3 on GPU v3:\t", vec3_GPU_v3, N);
    if (equal_vectors(vec3_CPU, vec3_GPU_v3, N)) printf("Obliczenia GPU v3 poprawne.\n");
    printf("Czas wykonania GPU v3:\t %f ms\n", calc_t_GPU_v3);

    // free memory on GPU
    CUDA_status = cudaFree(dev_vec1); catchException(CUDA_status);
    CUDA_status = cudaFree(dev_vec2); catchException(CUDA_status);
    CUDA_status = cudaFree(dev_vec3_v1); catchException(CUDA_status);
    CUDA_status = cudaFree(dev_vec3_v2); catchException(CUDA_status);
    CUDA_status = cudaFree(dev_vec3_v3); catchException(CUDA_status);

    exit(EXIT_SUCCESS);
}

__global__
void add_vec_GPU(int* vec1, int* vec2, int* vec3, size_t n) {

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

int equal_vectors(int* vec1, int* vec2, size_t n) {
    int i;
    for (i = 0; i < n; i++) {
        if (vec1[i] != vec2[i])
            return 0;
    }
    return 1;
}

void catchException(cudaError_t status) {
    if (status != cudaSuccess) {
        printf("%s\n", cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}