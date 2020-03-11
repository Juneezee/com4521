#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define N 65536
#define THREADS_PER_BLOCK 128
#define PUMP_RATE 2

#define READ_BYTES N*(2*4)  //2 reads of 4 bytes (a and b)
#define WRITE_BYTES N*(4*4) //4 write of 4 bytes (to c)

__device__ int d_a[N];
__device__ int d_b[N];
__device__ int d_c[N];

void checkCUDAError(const char *);
void random_ints(int *a);

__global__ void vectorAdd(int *a, int *b, int *c, int max) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    d_c[i] = d_a[i] + d_b[i];
}

int main(void) {
    const unsigned int size = N * sizeof(int);
    cudaEvent_t start, stop;
    float milliseconds = 0;
    int device_count = 0;
    double theoretical_bw = 0;

    cudaGetDeviceCount(&device_count);
    if (device_count > 0) {
        cudaSetDevice(0);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);

        // Calculate in GB/s
        theoretical_bw = deviceProp.memoryClockRate * PUMP_RATE * (deviceProp.memoryBusWidth / 8.0) / 1e6;
    }

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Alloc space for host copies of a, b, c and setup input values
    int *a = (int *)malloc(size);
    random_ints(a);
    int *b = (int *)malloc(size);
    random_ints(b);
    int *c = (int *)malloc(size);

    // Copy inputs to device
    cudaMemcpyToSymbol(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_b, b, size, cudaMemcpyHostToDevice);

    // Launch add() kernel on GPU
    cudaEventRecord(start);
    vectorAdd << <N / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (d_a, d_b, d_c, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds, start, stop);
    const double measure_bw = (READ_BYTES + WRITE_BYTES) / (milliseconds * 1e6);

    // Copy result back to host
    cudaMemcpyFromSymbol(c, d_c, size);

    // Destroy event
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Cleanup
    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    printf("Execution time is %f ms\n", milliseconds);
    printf("Theoretical Bandwidth is %f GB/s\n", theoretical_bw);
    printf("Measured Bandwidth is %f GB/s\n", measure_bw);
    return 0;
}

void random_ints(int *a) {
    for (unsigned int i = 0; i < N; i++) {
        a[i] = rand();
    }
}
