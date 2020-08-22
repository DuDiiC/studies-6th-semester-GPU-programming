#include <cstdlib>
#include <cstdio>
#include <cmath>

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

#define N 100

struct square { 
	__host__ __device__ int operator()(int x) { 
		return x * x; 
	} 
};

double CPU_computing(long* x, long* y, size_t n);
double GPU_computing(long* x, long* y, size_t n);

void init_values(long* x, long* y, size_t n);
void print_vec(const char* label, long* vec, size_t n);

int main() {

	long h_x[N],
		 h_y[N];
	double CPU_result;
	double GPU_result;

	init_values(h_x, h_y, N);

	CPU_result = CPU_computing(h_x, h_y, N);
	GPU_result = GPU_computing(h_x, h_y, N);

	//print_vec("vec1:\t", h_x, N);
	//print_vec("vec2:\t", h_y, N);
	printf("Obliczenia na CPU:\t%f\n", CPU_result);
	printf("Obliczenia na GPU:\t%f\n", GPU_result);

	exit(EXIT_SUCCESS);
}

double CPU_computing(long* x, long* y, size_t n) {
	int i;
	long* tmp = (long*)malloc(sizeof(tmp[0]) * n);
	if (!tmp) {
		printf("Unexpected exception while allocating memory for pointer.\n");
		exit(EXIT_FAILURE);
	}
	
	// TMP <- X - Y
	for (i = 0; i < n; i++) {
		tmp[i] = x[i] - y[i];
	}

	// Z <- euc_norm(TMP)
	// sum
	int tmp2 = 0;
	for (i = 0; i < n; i++) {
		tmp2 += tmp[i] * tmp[i];
	}
	free(tmp);
	// sqrt
	return sqrt(tmp2);
}

double GPU_computing(long* x, long* y, size_t n) {
	
	thrust::device_vector<long> d_x(x, x + n);
	thrust::device_vector<long> d_y(y, y + n);
	thrust::device_vector<long> tmp(n);

	// TMP <- X - Y
	thrust::transform(d_x.begin(), d_x.end(), d_y.begin(), tmp.begin(), thrust::minus<int>());

	// RED <- euc_norm(TMP)
	return sqrt(thrust::transform_reduce(tmp.begin(), tmp.end(), square(), 1, thrust::plus<int>()));
}

void init_values(long* x, long* y, size_t n) {
	int i;
	for (i = 0; i < n; i++) {
		x[i] = i * i;
		y[i] = -i;
	}
}

void print_vec(const char* label, long* vec, size_t n) {
	int i;
	printf("%s [", label);
	for (i = 0; i < n - 1; i++)
		printf("%4d, ", vec[i]);
	printf("%4d]\n", vec[n - 1]);
}