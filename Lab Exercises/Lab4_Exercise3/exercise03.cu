#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define N 2048
#define M 1024
#define THREADS_PER_BLOCK 256
#define SQRT_THREADS_PER_BLOCK sqrt(THREADS_PER_BLOCK)

void checkCUDAError(const char *);
void random_ints(int *a);
void matrixAddCPU(int *a, int *b, int *c);
int validate(int *c, int *c_ref);

__global__ void matrixAdd(int *a, int *b, int *c) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < N && y < M) {
        const int i = y * N + x;
        c[i] = a[i] + b[i];
    }
}

int main() {
    int *a, *b, *c, *c_ref; // host copies of a, b, c
    int *d_a, *d_b, *d_c; // device copies of a, b, c
    int errors;
    unsigned int size = N * M * sizeof(int);

    // Alloc space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);
    checkCUDAError("CUDA malloc");

    // Alloc space for host copies of a, b, c and setup input values
    a = (int *)malloc(size);
    random_ints(a);
    b = (int *)malloc(size);
    random_ints(b);
    c = (int *)malloc(size);
    c_ref = (int *)malloc(size);

    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    checkCUDAError("CUDA memcpy");

    // Launch add() kernel on GPU
    unsigned int thread_width = (unsigned int)SQRT_THREADS_PER_BLOCK;
    dim3 threadsPerBlock(thread_width, thread_width, 1);

    unsigned int block_width = (unsigned int)ceil((double)N / thread_width);
    unsigned int block_height = (unsigned int)ceil((double)M / thread_width);
    dim3 blocksPerGrid(block_width, block_height, 1);

    matrixAdd << <blocksPerGrid, threadsPerBlock >> > (d_a, d_b, d_c);
    checkCUDAError("CUDA kernel");

    matrixAddCPU(a, b, c_ref);

    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    checkCUDAError("CUDA memcpy");

    errors = validate(c, c_ref);
    printf("GPU result has %d errors\n", errors);

    // Cleanup
    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    checkCUDAError("CUDA cleanup");

    return 0;
}

void checkCUDAError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void random_ints(int *a) {
    for (unsigned int x = 0; x < N; x++) {
        for (unsigned int y = 0; y < M; y++) {
            a[y * N + x] = rand();
        }
    }
}

void matrixAddCPU(int *a, int *b, int *c_ref) {
    for (unsigned int x = 0; x < N; x++) {
        for (unsigned int y = 0; y < M; y++) {
            int i = y * N + x;
            c_ref[i] = a[i] + b[i];
        }
    }
}

int validate(int *c, int *c_ref) {
    int errors = 0;

    for (unsigned int x = 0; x < N; x++) {
        for (unsigned int y = 0; y < M; y++) {
            unsigned int i = y * N + x;
            if (c[i] != c_ref[i]) {
                errors++;
                fprintf(stderr, "ERROR at index %u: GPU result %d != CPU value of %d\n", i, c[i], c_ref[i]);
            }
        }
    }

    return errors;
}
