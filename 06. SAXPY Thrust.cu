// KOMENTARZ:
//
// Testowano na:
//      - CPU: Intel Core i7-7700HQ (2.8 - 3.8 GHz, 6MB cache)
//      - GPU: NVIDIA GeFore RTX 1050 (2048 MB GDDR5) 
//
// Najszybciej algorytm wykoywal sie przy uzyciu standardowych funcji CUDA,
// natomiast jesli chodzi o roznice miedzy wykonywaniem funkcji SAXPY przy
// uzyciu biblioteki Thrust, duzo szybsze okazalo sie uzycie funktora, w 
// porownaniu z naiwnym wywolywaniem algorytmu krok po kroku.
//
// ===================================================================

#include <iostream>
#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <cuda_runtime_api.h>

#define N 1000
#define ALFA 4

cudaError_t CUDA_status = cudaSuccess;

// functor code from https://github.com/thrust/thrust/blob/master/examples/saxpy.cu
struct saxpy_functor : public thrust::binary_function<int, int, int> {
	const int a;

	saxpy_functor(int _a) : a(_a) {}

	__host__ __device__
		int operator()(const int& x, const int& y) const {
		return a * x + y;
	}
};

void saxpy_CPU(int alfa, int* x, int* y, int* z, size_t n);
__global__ void saxpy_CUDA(int alfa, int* x, int* y, int* z, size_t n);
void saxpy_on_CUDA(int alfa, int* x, int* y, int* z);
void saxpy_on_Thrust_v1(int alfa, int* x, int* y, int* z, size_t n);
void saxpy_on_Thrust_v2(int alfa, int* x, int* y, int* z, size_t n);
void init_values(int* x, int* y, size_t n);
int equal_vectors(int* vec1, int* vec2, size_t n);
void print_vec(const char* label, int* vec, size_t n);
void catchException(cudaError_t status);

int main() {

	int h_x[N],
		h_y[N],
		h_z[N];

	init_values(h_x, h_y, N);
	//print_vec("vec1:\t", h_x, N);
	//print_vec("vec2:\t", h_y, N);
	saxpy_CPU(ALFA, h_x, h_y, h_z, N);

	init_values(h_x, h_y, N);
	saxpy_on_CUDA(ALFA, h_x, h_y, h_z);
	
	init_values(h_x, h_y, N);
	saxpy_on_Thrust_v1(ALFA, h_x, h_y, h_z, N);
	
	init_values(h_x, h_y, N);
	saxpy_on_Thrust_v2(ALFA, h_x, h_y, h_z, N);

	return 0;
}

void saxpy_CPU(int alfa, int* x, int* y, int* z, size_t n) {
	int i;
	for (i = 0; i < n; i++) {
		z[i] = alfa * x[i] + y[i];
	}
}

__global__
void saxpy_CUDA(int alfa, int* x, int* y, int* z, size_t n) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	while (tid < n) {
		z[tid] = alfa * x[tid] + y[tid];
		tid += gridDim.x * blockDim.x;
	}
}

void saxpy_on_CUDA(int alfa, int* x, int* y, int* z) {

	int h_z[N];

	int *d_x,
		*d_y,
		*d_z;

	int threads_per_block = 256;
	int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

	float calc_t;
	cudaEvent_t start, stop;
	CUDA_status = cudaEventCreate(&start); catchException(CUDA_status);
	CUDA_status = cudaEventCreate(&stop); catchException(CUDA_status);

	// start of execution time measurement
	CUDA_status = cudaEventRecord(start, 0);
	catchException(CUDA_status);

	// memory allocation on GPU
	CUDA_status = cudaMalloc((void**)&d_x, N * sizeof(d_x[0]));
	catchException(CUDA_status);
	CUDA_status = cudaMalloc((void**)&d_y, N * sizeof(d_y[0]));
	catchException(CUDA_status);	
	CUDA_status = cudaMalloc((void**)&d_z, N * sizeof(d_z[0]));
	catchException(CUDA_status);

	// copy data from CPU to GPU
	CUDA_status = cudaMemcpy(d_x, x, N * sizeof(d_x[0]), cudaMemcpyHostToDevice);
	catchException(CUDA_status);
	CUDA_status = cudaMemcpy(d_y, y, N * sizeof(d_y[0]), cudaMemcpyHostToDevice);
	catchException(CUDA_status);

	// calculate on GPU
	saxpy_CUDA << < blocks_per_grid, threads_per_block>> > (ALFA, d_x, d_y, d_z, N);

	// copy result from GPU toCPU
	CUDA_status = cudaMemcpy(h_z, d_z, N * sizeof(h_z[0]), cudaMemcpyDeviceToHost);
	catchException(CUDA_status);

	// end of execution time measurement
	CUDA_status = cudaEventRecord(stop);
	catchException(CUDA_status);
	CUDA_status = cudaEventSynchronize(stop);
	catchException(CUDA_status);

	// calculate execution time
	CUDA_status = cudaEventElapsedTime(&calc_t, start, stop);
	catchException(CUDA_status);

	// print results
	printf("==== CUDA ====\n");
	//print_vec("wynik:\t", h_z, N);
	if (equal_vectors(h_z, z, N)) printf("Obliczenia poprawne.\n");
	printf("Czas wykonania:\t%f ms\n", calc_t);
}

void saxpy_on_Thrust_v1(int alfa, int*x, int *y, int *z, size_t n) {
	
	float calc_t;
	cudaEvent_t start, stop;
	CUDA_status = cudaEventCreate(&start); catchException(CUDA_status);
	CUDA_status = cudaEventCreate(&stop); catchException(CUDA_status);

	CUDA_status = cudaEventRecord(start, 0);
	catchException(CUDA_status);

	thrust::device_vector<int> X(x, x + n);
	thrust::device_vector<int> Y(y, y + n);
	thrust::device_vector<int> temp(X.size());

	// temp <- A
	thrust::fill(temp.begin(), temp.end(), alfa);

	// temp <- A * X
	thrust::transform(X.begin(), X.end(), temp.begin(), temp.begin(), thrust::multiplies<int>());

	// Y <- A * X + Y
	thrust::transform(temp.begin(), temp.end(), Y.begin(), Y.begin(), thrust::plus<int>());

	CUDA_status = cudaEventRecord(stop);
	catchException(CUDA_status);
	CUDA_status = cudaEventSynchronize(stop);
	catchException(CUDA_status);

	CUDA_status = cudaEventElapsedTime(&calc_t, start, stop);
	catchException(CUDA_status);

	int* y_d_ptr = thrust::raw_pointer_cast(&Y[0]);
	int y_ptr[N];
	cudaMemcpy(y_ptr, y_d_ptr, n * sizeof(y_ptr[0]), cudaMemcpyDeviceToHost);

	// print results
	printf("==== Thrust V1 ====\n");
	//print_vec("wynik:\t", y_ptr, N);
	if (equal_vectors(z, y_ptr, N)) printf("Obliczenia poprawne.\n");
	printf("Czas wykonania:\t%f ms\n", calc_t);
}

void saxpy_on_Thrust_v2(int alfa, int* x, int* y, int* z, size_t n) {
	
	float calc_t;
	cudaEvent_t start, stop;
	CUDA_status = cudaEventCreate(&start); catchException(CUDA_status);
	CUDA_status = cudaEventCreate(&stop); catchException(CUDA_status);

	CUDA_status = cudaEventRecord(start, 0);
	catchException(CUDA_status);

	thrust::device_vector<int> X(x, x + n);
	thrust::device_vector<int> Y(y, y + n);
	thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(), saxpy_functor(alfa));

	CUDA_status = cudaEventRecord(stop);
	catchException(CUDA_status);
	CUDA_status = cudaEventSynchronize(stop);
	catchException(CUDA_status);

	CUDA_status = cudaEventElapsedTime(&calc_t, start, stop);
	catchException(CUDA_status);

	int* y_d_ptr = thrust::raw_pointer_cast(&Y[0]);
	int y_ptr[N];
	cudaMemcpy(y_ptr, y_d_ptr, n * sizeof(y_ptr[0]), cudaMemcpyDeviceToHost);

	// print results
	printf("==== Thrust V2 ====\n");
	//print_vec("wynik:\t", y_ptr, N);
	if (equal_vectors(z, y_ptr, N)) printf("Obliczenia poprawne.\n");
	printf("Czas wykonania:\t%f ms\n", calc_t);
}

void init_values(int* x, int* y, size_t n) {
	int i;
	for (i = 0; i < n; i++) {
		x[i] = i * i;
		y[i] = -i;
	}
}

int equal_vectors(int* vec1, int* vec2, size_t n) {
	int i;
	for (i = 0; i < n; i++) {
		if (vec1[i] != vec2[i])
			return 0;
	}
	return 1;
}

void print_vec(const char* label, int* vec, size_t n) {
	int i;
	printf("%s [", label);
	for (i = 0; i < n - 1; i++)
		printf("%4d, ", vec[i]);
	printf("%4d]\n", vec[n - 1]);
}

void catchException(cudaError_t status)  {
	if (status != cudaSuccess) {
		printf("%s\n", cudaGetErrorString(status));
		exit(EXIT_FAILURE);
	}
}