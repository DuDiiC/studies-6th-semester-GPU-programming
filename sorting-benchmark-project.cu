// Author: Maciej Dudek
// 27.08.2020
//
// sources:
// materials from "GPU programming course" [ https://plas.mat.umk.pl ]
// CUDA basics from NVIDIA [ https://www.nvidia.com/docs/IO/116711/sc11-cuda-c-basics.pdf ]
// presentation about parallel sorting algorithms [ http://aragorn.pb.bialystok.pl/~wkwedlo/PC10.pdf ]
// theory about merge sort [ https://www.geeksforgeeks.org/merge-sort/ ]
// theory about quick sort [ https://www.geeksforgeeks.org/quick-sort/ ]
// example of sort implementation in THRUST [ https://stackoverflow.com/questions/30903109/how-do-you-build-the-example-cuda-thrust-device-sort ]
// chrono documentation for measuring execution time [ https://en.cppreference.com/w/cpp/chrono ]
//
// Tested on:
//      - CPU: Intel Core i7-7700HQ (2.8 - 3.8 GHz, 6MB cache)
//      - GPU: NVIDIA GeFore RTX 1050 (2048 MB GDDR5)

// c libs
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// cpp libs
#include <chrono>

// cuda
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h" // fix Visual Intellisense problem with threadIdx for example

// thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

#define CALLS_NUMBER 5
#define TALKATIVE 0
#define THREADS_PER_BLOCK 1024

// ============== PREPARE DATA ==============
void create_random_array(int* array_to_sort, size_t size);
void create_sorted_array(int* array_to_sort, size_t size);
void create_revert_sorted_array(int* array_to_sort, size_t size);

// ============== ALGORITHMS ==============
// quicksort CPU
void CPU_quicksort(int* array_to_sort, size_t size);
void qs(int* array, int a_index, int b_index);

// mergesort CPU
void CPU_mergesort(int* array_to_sort, size_t size);
void ms(int* array, int* tmp_array, int a_index, int b_index);
void merge(int* array, int* tmp_array, int a_index, int middle_index, int b_index);

// ranksort CPU
void CPU_ranksort(int* array_to_sort, size_t size);
void CPU_rs(int* array_to_sort, int* sorted_array, int* ranks_array, size_t size);
void rs_fix_problem_with_repetitions(int* sorted_array, size_t size);

// ranksort GPU
void GPU_ranksort(int* array_to_sort, size_t size);
__global__ void GPU_rs(int* array_to_sort, int* ranks_array, size_t size);
__global__ void GPU_rs_set_ranks(int* array_to_sort, int* ranks_array, int* sorted_array, size_t size);

// odd-even sort
void GPU_oddevensort(int* array_to_sort, size_t size);
__global__ void GPU_oes_odd(int* array_to_sort, size_t size);
__global__ void GPU_oes_even(int* array_to_sort, size_t size);

// thrust implementation sort
void GPU_thrust_sort(int* array_to_sort, size_t size);

// ============== UTILS FUNCTIONS ==============
void swap(int* array, int a_position, int b_position);
bool check_sorting(int* array, size_t size);
void print_array(char* label, int* array, size_t size);
void catch_cuda_exception(cudaError_t status);

int main()
{
    int* array_to_sort;

    for (int i = 10; i <= 1000000; i*=10) {
        array_to_sort = (int*)malloc(sizeof(array_to_sort[0]) * i);
        if (!array_to_sort) {
            perror("malloc");
            exit(EXIT_FAILURE);
        }

        printf("\n\n================================= ARRAY SIZE: %d  =================================\n\n", i);
        printf("\n============ RANDOM ARRAY ============\n\n");
        create_random_array(array_to_sort, i);
        if (i <= 10000) CPU_quicksort(array_to_sort, i); // stackoverflow exception
        CPU_mergesort(array_to_sort, i);
        if (i <= 10000) CPU_ranksort(array_to_sort, i); // too long
        if (i <= 100000) GPU_ranksort(array_to_sort, i); // too long
        if (i <= 100000) GPU_oddevensort(array_to_sort, i); // too long
        GPU_thrust_sort(array_to_sort, i);

        printf("\n============ SORTED ARRAY ============\n\n");
        create_sorted_array(array_to_sort, i);
        if (i <= 10000) CPU_quicksort(array_to_sort, i);
        CPU_mergesort(array_to_sort, i);
        if (i <= 10000) CPU_ranksort(array_to_sort, i);
        if (i <= 100000) GPU_ranksort(array_to_sort, i);
        if (i <= 100000) GPU_oddevensort(array_to_sort, i);
        GPU_thrust_sort(array_to_sort, i);

        printf("\n============ REVERT SORTED ARRAY ============\n\n");
        create_revert_sorted_array(array_to_sort, i);
        if (i <= 10000) CPU_quicksort(array_to_sort, i);
        CPU_mergesort(array_to_sort, i);
        if (i <= 10000) CPU_ranksort(array_to_sort, i);
        if (i <= 100000) GPU_ranksort(array_to_sort, i);
        if (i <= 100000) GPU_oddevensort(array_to_sort, i);
        GPU_thrust_sort(array_to_sort, i);

        free(array_to_sort);
    }

    exit(EXIT_SUCCESS);
}

// ============== PREPARE DATA ==============
void create_random_array(int* array_to_sort, size_t size)
{
    int i;
    srand(time(NULL));
    for (i = 0; i < size; i++) {
        array_to_sort[i] = (int)rand();
    }
}

void create_sorted_array(int* array_to_sort, size_t size) 
{
    int i;
    for (i = 0; i < size; i++) {
        array_to_sort[i] = 2 * i;
    }
}

void create_revert_sorted_array(int* array_to_sort, size_t size)
{
    int i;
    for (i = 0; i < size; i++) {
        array_to_sort[i] = -2 * i;
    }
}

// ============== ALGORITHMS ==============
// quicksort CPU
void CPU_quicksort(int* array_to_sort, size_t size)
{
    int* array = (int*)malloc(sizeof(array_to_sort[0]) * size);
    if (!array) {
        perror("malloc");
        exit(EXIT_FAILURE);
    }
    double sort_time_ms_average = 0;

    printf("Quicksort CPU:\n");
    for (int i = 1; i <= CALLS_NUMBER; i++) {
        memcpy(array, array_to_sort, size);

        auto start = std::chrono::steady_clock::now();

        qs(array, 0, size - 1);

        auto end = std::chrono::steady_clock::now();
        auto sort_time_ms = std::chrono::duration<double>(end - start).count() * 1000.0;

        if (check_sorting(array, size)) {
            if (TALKATIVE) printf("\t%d:\t\t%f ms\n", i, sort_time_ms);
        }
        else {
            printf("%d: \t\tWrong sorting!\n", i);
        }
        sort_time_ms_average += sort_time_ms;
    }
    printf("\taverage (%d times sorted):\t%f ms\n", CALLS_NUMBER, sort_time_ms_average / CALLS_NUMBER);

    free(array);
}

void qs(int* array, int a_index, int b_index)
{
    if (a_index >= b_index) return;
    
    int pivot_value = array[b_index];
    int border_position = a_index - 1;
    int i = a_index;

    while (i < b_index) {
        if (array[i] < pivot_value) {
            border_position++;
            if (border_position != i) {
                swap(array, border_position, i);
            }
        }
        i++;
    }

    border_position++;
    if (border_position != b_index) {
        swap(array, border_position, i);
    }

    qs(array, a_index, border_position - 1);
    qs(array, border_position + 1, b_index);
}

// mergesort CPU
void CPU_mergesort(int* array_to_sort, size_t size)
{
    int* array = (int*)malloc(sizeof(array_to_sort[0]) * size);
    int* tmp_array = (int*)malloc(sizeof(array_to_sort[0]) * size);
    if (!array  || !tmp_array) {
        perror("malloc");
        exit(EXIT_FAILURE);
    }
    double sort_time_ms_average = 0;

    printf("Mergesort CPU:\n");
    for (int i = 1; i <= CALLS_NUMBER; i++) {
        memcpy(array, array_to_sort, size);

        auto start = std::chrono::steady_clock::now();

        ms(array, tmp_array, 0, size - 1);

        auto end = std::chrono::steady_clock::now();
        auto sort_time_ms = std::chrono::duration<double>(end - start).count() * 1000.0;

        if (check_sorting(array, size)) {
            if (TALKATIVE) printf("\t%d:\t\t%f ms\n", i, sort_time_ms);
        }
        else {
            printf("%d: \t\tWrong sorting!\n", i);
        }
        sort_time_ms_average += sort_time_ms;
    }
    printf("\taverage (%d times sorted):\t%f ms\n", CALLS_NUMBER, sort_time_ms_average / CALLS_NUMBER);

    free(array);
    free(tmp_array);
}

void ms(int* array, int* tmp_array, int a_index, int b_index)
{
    if (a_index < b_index) {
        int middle_index = (a_index + b_index) / 2;
        ms(array, tmp_array, a_index, middle_index);
        ms(array, tmp_array, middle_index + 1, b_index);
        merge(array, tmp_array, a_index, middle_index, b_index);
    }
}

void merge(int* array, int* tmp_array, int a_index, int middle_index, int b_index)
{
    int i;
    for (i = a_index; i <= b_index; i++) {
        tmp_array[i] = array[i];
    }

    int tmp_1_index = a_index;
    int tmp_2_index = middle_index + 1;
    int current_index = a_index;

    while (tmp_1_index <= middle_index && tmp_2_index <= b_index) {
        if (tmp_array[tmp_1_index] <= tmp_array[tmp_2_index]) {
            array[current_index] = tmp_array[tmp_1_index];
            tmp_1_index++;
        }
        else {
            array[current_index] = tmp_array[tmp_2_index];
            tmp_2_index++;
        }
        current_index++;
    }

    while (tmp_1_index <= middle_index) {
        array[current_index] = tmp_array[tmp_1_index];
        current_index++;
        tmp_1_index++;
    }
}

// ranksort CPU
void CPU_ranksort(int* array_to_sort, size_t size)
{
    int* sorted_array = (int*)malloc(sizeof(array_to_sort[0]) * size);
    
    int* ranks_array = (int*)malloc(sizeof(int) * size);
    if (!sorted_array || !ranks_array) {
        perror("malloc");
        exit(EXIT_FAILURE);
    }
    double sort_time_ms_average = 0;

    printf("Ranksort CPU:\n");
    for (int i = 1; i <= CALLS_NUMBER; i++) {
        memset(sorted_array, 0, sizeof(sorted_array[0]) * size);

        auto start = std::chrono::steady_clock::now();

        CPU_rs(array_to_sort, sorted_array, ranks_array, size);

        auto end = std::chrono::steady_clock::now();
        auto sort_time_ms = std::chrono::duration<double>(end - start).count() * 1000.0;

        rs_fix_problem_with_repetitions(sorted_array, size);

        if (check_sorting(sorted_array, size)) {
            if (TALKATIVE) printf("\t%d:\t\t%f ms\n", i, sort_time_ms);
        }
        else {
            printf("%d: \t\tWrong sorting!\n", i);
        }
        sort_time_ms_average += sort_time_ms;
    }
    printf("\taverage (%d times sorted):\t%f ms\n", CALLS_NUMBER, sort_time_ms_average / CALLS_NUMBER);

    free(sorted_array); free(ranks_array);
}

void CPU_rs(int* array_to_sort, int* sorted_array, int* ranks_array, size_t size)
{
    int i, j;
    for (i = 0; i < size; i++) {
        int rank = 0;
        for (j = 0; j < size; j++) {
            if (i != j) {
                if (array_to_sort[i] > array_to_sort[j]) {
                    rank++;
                }
            }
        }
        ranks_array[i] = rank;
    }
    for (i = 0; i < size; i++) {
        sorted_array[ranks_array[i]] = array_to_sort[i];
    }
}

void rs_fix_problem_with_repetitions(int* sorted_array, size_t size)
{
    int i;
    for (i = 1; i < size; i++) {
        if (sorted_array[i] == 0) {
            sorted_array[i] = sorted_array[i - 1];
        }
    }
}

// ranksort GPU
void GPU_ranksort(int* array_to_sort, size_t size)
{
    int* sorted_array = (int*)malloc(sizeof(sorted_array[0]) * size);
    int* ranks_array = (int*)malloc(sizeof(int) * size);
    if (!sorted_array || !ranks_array) {
        perror("malloc");
        exit(EXIT_FAILURE);
    }

    int* array_d;
    int* sorted_array_d;
    int* ranks_array_d;
    cudaError_t cuda_status = cudaSuccess;
    int threads_per_block = THREADS_PER_BLOCK;
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;

    cuda_status = cudaMalloc((void**)&array_d, sizeof(array_d[0]) * size);
    catch_cuda_exception(cuda_status);
    cuda_status = cudaMalloc((void**)&sorted_array_d, sizeof(sorted_array_d[0]) * size);
    catch_cuda_exception(cuda_status);
    cuda_status = cudaMalloc((void**)&ranks_array_d, sizeof(ranks_array_d[0]) * size);
    catch_cuda_exception(cuda_status);

    float sort_time_ms_average = 0.0;

    printf("Ranksort GPU:\n");
    for (int i = 1; i <= CALLS_NUMBER; i++) {
        cuda_status = cudaMemcpy(array_d, array_to_sort, sizeof(array_to_sort[0]) * size, cudaMemcpyHostToDevice);
        catch_cuda_exception(cuda_status);
        cuda_status = cudaMemcpy(ranks_array_d, ranks_array, sizeof(ranks_array[0]) * size, cudaMemcpyHostToDevice);
        catch_cuda_exception(cuda_status);

        float sort_time_ms;
        cudaEvent_t start, stop;
        cuda_status = cudaEventCreate(&start); catch_cuda_exception(cuda_status);
        cuda_status = cudaEventCreate(&stop); catch_cuda_exception(cuda_status);

        cuda_status = cudaEventRecord(start); catch_cuda_exception(cuda_status);

        GPU_rs << <blocks_per_grid, threads_per_block >> > (array_d, ranks_array_d, size);
        GPU_rs_set_ranks << <blocks_per_grid, threads_per_block >> > (array_d, ranks_array_d, sorted_array_d, size);

        cuda_status = cudaEventRecord(stop); catch_cuda_exception(cuda_status);
        cuda_status = cudaEventSynchronize(stop); catch_cuda_exception(cuda_status);
        cuda_status = cudaEventElapsedTime(&sort_time_ms, start, stop); catch_cuda_exception(cuda_status);

        cuda_status = cudaMemcpy(sorted_array, sorted_array_d, sizeof(sorted_array_d[0]) * size, cudaMemcpyDeviceToHost);
        catch_cuda_exception(cuda_status);
        cuda_status = cudaMemcpy(ranks_array, ranks_array_d, sizeof(ranks_array_d[0]) * size, cudaMemcpyDeviceToHost);

        rs_fix_problem_with_repetitions(sorted_array, size);

        if (check_sorting(sorted_array, size)) {
            if (TALKATIVE) printf("\t%d:\t\t%f ms\n", i, sort_time_ms);
        }
        else {
            printf("%d: \t\tWrong sorting!\n", i);
        }
        sort_time_ms_average += sort_time_ms;
    }
    printf("\taverage (%d times sorted):\t%f ms\n", CALLS_NUMBER, sort_time_ms_average / CALLS_NUMBER);

    cudaFree(array_d); cudaFree(sorted_array_d); cudaFree(ranks_array_d);
    free(sorted_array); free(ranks_array);
}

__global__ void GPU_rs(int* array_to_sort, int* ranks_array, size_t size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < size) {
        int i;
        int rank = 0;
        for (i = 0; i < size; i++) {
            if (i != tid) {
                if (array_to_sort[tid] > array_to_sort[i]) {
                    rank++;
                }
            }
        }
        ranks_array[tid] = rank;
        tid += gridDim.x * blockDim.x;
    }
}

__global__ void GPU_rs_set_ranks(int* array_to_sort, int* ranks_array, int* sorted_array, size_t size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < size) {
        sorted_array[ranks_array[tid]] = array_to_sort[tid];
        tid += gridDim.x * blockDim.x;
    }
    return;
}

// odd-even sort GPU
void GPU_oddevensort(int* array_to_sort, size_t size)
{
    int* array = (int*)malloc(sizeof(array_to_sort[0]) * size);
    if (!array) {
        perror("malloc");
        exit(EXIT_FAILURE);
    }

    memcpy(array, array_to_sort, size);

    int* array_d;
    cudaError_t cuda_status = cudaSuccess;
    int threads_per_block = THREADS_PER_BLOCK;
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;

    cuda_status = cudaMalloc((void**)&array_d, sizeof(array_d[0]) * size);
    catch_cuda_exception(cuda_status);

    float sort_time_ms_average = 0.0;

    printf("Odd-even sort GPU:\n");
    for (int i = 0; i < CALLS_NUMBER; i++) {
        cuda_status = cudaMemcpy(array_d, array, sizeof(array[0]) * size, cudaMemcpyHostToDevice);
        catch_cuda_exception(cuda_status);

        float sort_time_ms;
        cudaEvent_t start, stop;
        cuda_status = cudaEventCreate(&start); catch_cuda_exception(cuda_status);
        cuda_status = cudaEventCreate(&stop); catch_cuda_exception(cuda_status);

        cuda_status = cudaEventRecord(start); catch_cuda_exception(cuda_status);

        for (int j = 0; j < size; j++) {
            if (j % 2 == 1) { // odd step
                GPU_oes_odd << <blocks_per_grid, threads_per_block >> > (array_d, size);
            }
            else { // even step
                GPU_oes_even << <blocks_per_grid, threads_per_block >> > (array_d, size);
            }
        }

        cuda_status = cudaEventRecord(stop); catch_cuda_exception(cuda_status);
        cuda_status = cudaEventSynchronize(stop); catch_cuda_exception(cuda_status);
        cuda_status = cudaEventElapsedTime(&sort_time_ms, start, stop); catch_cuda_exception(cuda_status);

        cuda_status = cudaMemcpy(array, array_d, sizeof(array_d[0]) * size, cudaMemcpyDeviceToHost);
        catch_cuda_exception(cuda_status);

        if (check_sorting(array, size)) {
            if (TALKATIVE) printf("\t%d:\t\t%f ms\n", i, sort_time_ms);
        }
        else {
            printf("%d: \t\tWrong sorting!\n", i);
        }
        sort_time_ms_average += sort_time_ms;
    }
    printf("\taverage (%d times sorted):\t%f ms\n", CALLS_NUMBER, sort_time_ms_average / CALLS_NUMBER);

    cudaFree(array_d);
    free(array);
}

__global__ void GPU_oes_odd(int* array_to_sort, size_t size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid % 2 == 1) {
        while (tid + 1 < size) {
            if (array_to_sort[tid] > array_to_sort[tid + 1]) {
                int tmp = array_to_sort[tid];
                array_to_sort[tid] = array_to_sort[tid + 1];
                array_to_sort[tid + 1] = tmp;
            }
            tid += gridDim.x * blockDim.x;
        }
    }
}

__global__ void GPU_oes_even(int* array_to_sort, size_t size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid % 2 == 0) {
        while (tid + 1 < size) {
            if (array_to_sort[tid] > array_to_sort[tid + 1]) {
                int tmp = array_to_sort[tid];
                array_to_sort[tid] = array_to_sort[tid + 1];
                array_to_sort[tid + 1] = tmp;
            }

            tid += gridDim.x * blockDim.x;
        }
    }
}

// thrust implementation sort
void GPU_thrust_sort(int* array_to_sort, size_t size)
{
    int* array = (int*)malloc(sizeof(array_to_sort[0]) * size);
    if (!array) {
        perror("malloc");
        exit(EXIT_FAILURE);
    }
    
    cudaError_t cuda_status = cudaSuccess;
    float sort_time_ms_average = 0.0;

    printf("Thrust GPU:\n");
    for (int i = 0; i < CALLS_NUMBER; i++) {
        memcpy(array, array_to_sort, size);

        thrust::device_vector<int> d_vec(array, array + size);

        float sort_time_ms;
        cudaEvent_t start, stop;
        cuda_status = cudaEventCreate(&start); catch_cuda_exception(cuda_status);
        cuda_status = cudaEventCreate(&stop); catch_cuda_exception(cuda_status);

        cuda_status = cudaEventRecord(start); catch_cuda_exception(cuda_status);

        thrust::sort(d_vec.begin(), d_vec.end());

        cuda_status = cudaEventRecord(stop); catch_cuda_exception(cuda_status);
        cuda_status = cudaEventSynchronize(stop); catch_cuda_exception(cuda_status);
        cuda_status = cudaEventElapsedTime(&sort_time_ms, start, stop); catch_cuda_exception(cuda_status);

        thrust::host_vector<int> h_vec = d_vec;
       
        for (int j = 0; j < size; j++) {
            array[j] = h_vec[j];
        }

        if (check_sorting(array, size)) {
            if (TALKATIVE) printf("\t%d:\t\t%f ms\n", i, sort_time_ms);
        }
        else {
            printf("%d: \t\tWrong sorting!\n", i);
        }
        sort_time_ms_average += sort_time_ms;
    }
    printf("\taverage (%d times sorted):\t%f ms\n", CALLS_NUMBER, sort_time_ms_average / CALLS_NUMBER);
        
    free(array);
}

// ============== UTILS FUNCTIONS ==============
void swap(int* array, int a_position, int b_position) 
{
    int tmp = array[a_position];
    array[a_position] = array[b_position];
    array[b_position] = tmp;
}

bool check_sorting(int* array, size_t size)
{
    int i = 0;
    for (i = 0; i < size - 1; i++) {
        if (!(array[i] <= array[i + 1])) {
            printf("\n!!! ERROR: %d > %d !!!\n\n", array[i], array[i + 1]);
            return false;
        }
    }
    return true;
}

void print_array(char* label, int* array, size_t size)
{
    int i;

    printf("%s: [", label);
    for (i = 0; i < size - 1; i++) {
        printf("%d, ", array[i]);
    }
    printf("%d]\n\n", array[size - 1]);
}

void catch_cuda_exception(cudaError_t status)
{
    if (status != cudaSuccess) {
        printf("%s\n", cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}