// KOMENTARZ:
//
// Testowano na:
//      - CPU: Intel Core i7-7700HQ (2.8 - 3.8 GHz, 6MB cache)
//      - GPU: NVIDIA GeFore RTX 1050 (2048 MB GDDR5) 
//
// Na malej ilosci uzywanej pamieci nie widac przewagi obliczen na karcie graficznej w stosunku do CPU. Prawdopodobnie czas
// kopiowania zdecydowanie przewaza nad czasem potrzebnym na wykonanie obliczen. Wieksza roznica moglaby byc zauwazona przy znaczaco
// wiekszych zbiorach danych, czy tez bardziej skomplikowanych obliczeniach (zauwazmy, ze dla N = 10 obliczenia na CPU s¹ zdecydowanie
// szybsze, jednak dla N = 100000 roznica ta jest juz znaczaco mniejsza). Roznica pomiedzy CPU oraz GPU moze byc rowniez spowodowana 
// obliczaniem czasu wykonania przy pomocy roznych bibliotek. 
// Widac natomiast (mala, jednak istotna) roznice w wykonaniu przy uzyciu alokowania pamiêci na zmienne, porownuj¹c alokacje
// na CPU przy uzyciu malloc() oraz przy uzyciu cudaMallocHost(), na korzysc wersji drugiej.

// ===================================================================

// std C libs
#include <stdio.h>
#include <stdlib.h>

// CUDA libs
#include <cuda_runtime_api.h>

// chrono for time tests on CPU
#include <chrono>

// vector size
#define N 100000

/// <summary>Adding vectors on GPU</summary>
/// <param name="vec1">- first vector</param>
/// <param name="vec2">- second vector</param>
/// <param name="vec3">- third vector with sum</param>
/// <param name="n">- vectors size</param>
__global__ void add_vec_GPU(int* vec1, int* vec2, int* vec3, size_t n);

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
/// If not, it terminates the program with an error CUDA_status.</summary>
/// <param name="CUDA_status">- function execution CUDA_status (correct is the same as <c>cudaSucces</c>)</param>
void catchException(cudaError_t status);

int main()
{
    cudaError_t CUDA_status = cudaSuccess;
    int i;

    int vec1[N];
    int vec2[N];

    // set start values
    for (i = 0; i < N; i++) {
        vec1[i] = i * i;
        vec2[i] = -i;
    }

    int *vec3_CPU, 
        *vec3_GPU_v1, 
        *vec3_GPU_v2;

    int *dev_vec1, 
        *dev_vec2, 
        *dev_vec3_v1, 
        *dev_vec3_v2;

    std::chrono::duration<double> calc_t_CPU;
    float calc_t_GPU_v1, 
          calc_t_GPU_v2;
    
    struct timespec start_time,
                    end_time;

    cudaEvent_t start_GPU_v1,
                start_GPU_v2,
                stop_GPU_v1,
                stop_GPU_v2;

    CUDA_status = cudaEventCreate(&start_GPU_v1); catchException(CUDA_status);
    CUDA_status = cudaEventCreate(&start_GPU_v2); catchException(CUDA_status);
    CUDA_status = cudaEventCreate(&stop_GPU_v1); catchException(CUDA_status);
    CUDA_status = cudaEventCreate(&stop_GPU_v2); catchException(CUDA_status);

    // COMPUTING ON CPU
    // ===================================================================
    
    // start of execution time measurement
    auto start = std::chrono::steady_clock::now();

    // malloc on CPU
    vec3_CPU = (int*)malloc(N * sizeof(vec3_CPU[0]));

    // computing on CPU
    add_vec_CPU(vec1, vec2, vec3_CPU, N);

    // end of execution time measurement (and convert time to milisec)
    auto end = std::chrono::steady_clock::now();
    calc_t_CPU = (end - start) * 1000;

    // COMPUTING ON GPU - VERSION WITH MALLOC
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    // ===================================================================

    // start of execution time measurement
    CUDA_status = cudaEventRecord(start_GPU_v1, 0);
    catchException(CUDA_status);

    // memory allocation on CPU v1 (with malloc)
    vec3_GPU_v1 = (int*)malloc(N * sizeof(vec3_GPU_v1[0]));

    // memory allocation on GPU v1
    CUDA_status = cudaMalloc((void**)&dev_vec1, N * sizeof(dev_vec1[0]));
    catchException(CUDA_status);
    CUDA_status = cudaMalloc((void**)&dev_vec2, N * sizeof(dev_vec2[0]));
    catchException(CUDA_status);
    CUDA_status = cudaMalloc((void**)&dev_vec3_v1, N * sizeof(dev_vec3_v1[0]));
    catchException(CUDA_status);

    // copy to GPU v1
    CUDA_status = cudaMemcpy(dev_vec1, vec1, N * sizeof(vec1[0]), cudaMemcpyHostToDevice);
    catchException(CUDA_status);
    CUDA_status = cudaMemcpy(dev_vec2, vec2, N * sizeof(vec2[0]), cudaMemcpyHostToDevice);
    catchException(CUDA_status);

    // computing on GPU v1
    add_vec_GPU << < blocksPerGrid, threadsPerBlock >> > (dev_vec1, dev_vec2, dev_vec3_v1, N);

    // copy result to CPU v1
    CUDA_status = cudaMemcpy(vec3_GPU_v1, dev_vec3_v1, N * sizeof(dev_vec3_v1[0]), cudaMemcpyDeviceToHost);
    catchException(CUDA_status);

    // end of execution time measurement
    CUDA_status = cudaEventRecord(stop_GPU_v1);
    catchException(CUDA_status);
    CUDA_status = cudaEventSynchronize(stop_GPU_v1);
    catchException(CUDA_status);

    // calculate execution time
    CUDA_status = cudaEventElapsedTime(&calc_t_GPU_v1, start_GPU_v1, stop_GPU_v1);
    catchException(CUDA_status);

    // COMPUTING ON GPU - VERSION WITH PINNED
    // ===================================================================

    // start of execution time measurement
    CUDA_status = cudaEventRecord(start_GPU_v2, 0);
    catchException(CUDA_status);

    // memory allocation on CPU v2 (not handling exception, becouse it could change execution time)
    cudaMallocHost((void**)&vec3_GPU_v2, N * sizeof(vec3_GPU_v2[0]));

    // memory allocation on GPU v2
    CUDA_status = cudaMalloc((void**)&dev_vec1, N * sizeof(dev_vec1[0]));
    catchException(CUDA_status);
    CUDA_status = cudaMalloc((void**)&dev_vec2, N * sizeof(dev_vec2[0]));
    catchException(CUDA_status);
    CUDA_status = cudaMalloc((void**)&dev_vec3_v2, N * sizeof(dev_vec3_v2[0]));
    catchException(CUDA_status);

    // copy to GPU v2 (not needed, but I do it to keep the same conditions)
    CUDA_status = cudaMemcpy(dev_vec1, vec1, N * sizeof(vec1[0]), cudaMemcpyHostToDevice);
    catchException(CUDA_status);
    CUDA_status = cudaMemcpy(dev_vec2, vec2, N * sizeof(vec2[0]), cudaMemcpyHostToDevice);
    catchException(CUDA_status);

    // computing on GPU v2
    add_vec_GPU << < blocksPerGrid, threadsPerBlock >> > (dev_vec1, dev_vec2, dev_vec3_v2, N);

    // copy result to CPU v2
    CUDA_status = cudaMemcpy(vec3_GPU_v2, dev_vec3_v2, N * sizeof(dev_vec3_v2[0]), cudaMemcpyDeviceToHost);
    catchException(CUDA_status);

    // end of execution time measurement
    CUDA_status = cudaEventRecord(stop_GPU_v2);
    catchException(CUDA_status);
    CUDA_status = cudaEventSynchronize(stop_GPU_v2);
    catchException(CUDA_status);

    // calculate execution time
    CUDA_status = cudaEventElapsedTime(&calc_t_GPU_v2, start_GPU_v2, stop_GPU_v2);
    catchException(CUDA_status);

    // PRINT RESULTS
    // ===================================================================
    
    //print_vec("vector 1:\t\t", vec1, N);
    //print_vec("vector 2:\t\t", vec2, N);
    ///printf("==========\n");
    //print_vec("vector 3 na CPU:\t", vec3_CPU, N);
    printf("Czas wykonania CPU:\t %f ms\n", calc_t_CPU);
    //print_vec("vector 3 na GPU v1:\t", vec3_GPU_v1, N);
    printf("Czas wykonania GPU v1:\t %f ms\n", calc_t_GPU_v1);
    //print_vec("vector 3 na GPU v2:\t", vec3_GPU_v2, N);
    printf("Czas wykonania GPU v2:\t %f ms\n", calc_t_GPU_v2);

    // CLEANING
    // ===================================================================
    
    // destroy events
    CUDA_status = cudaEventDestroy(start_GPU_v1); catchException(CUDA_status);
    CUDA_status = cudaEventDestroy(stop_GPU_v1); catchException(CUDA_status);

    // free memory on GPU
    CUDA_status = cudaFree(dev_vec1); catchException(CUDA_status);
    CUDA_status = cudaFree(dev_vec2); catchException(CUDA_status);
    CUDA_status = cudaFree(dev_vec3_v1); catchException(CUDA_status);
    CUDA_status = cudaFree(dev_vec3_v2); catchException(CUDA_status);

    // free memory on CPU
    free(vec3_CPU);
    free(vec3_GPU_v1);
    CUDA_status = cudaFreeHost(vec3_GPU_v2);
    catchException(CUDA_status);

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

void catchException(cudaError_t status) {
    if (status != cudaSuccess) {
        printf("%s\n", cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}