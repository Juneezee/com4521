#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define A_WIDTH 1024
#define A_HEIGHT 1024
#define B_WIDTH 1024
#define B_HEIGHT 1024
#define C_WIDTH B_WIDTH
#define C_HEIGHT A_HEIGHT

#define NUM_SUBS (A_WIDTH / BLOCK_SIZE)

__device__ float d_A[A_HEIGHT][A_WIDTH];
__device__ float d_B[B_HEIGHT][B_WIDTH];
__device__ float d_C[C_HEIGHT][C_WIDTH];

float h_A[A_HEIGHT][A_WIDTH];
float h_B[B_HEIGHT][B_WIDTH];
float h_C[C_HEIGHT][C_WIDTH];
float h_C_ref[C_HEIGHT][C_WIDTH];

int requiredSM(int TPB);
void matrixMulCPU(float A[A_HEIGHT][A_WIDTH], float B[B_HEIGHT][B_WIDTH], float C[C_HEIGHT][C_WIDTH]);
int matrixMulTest(float C[C_HEIGHT][C_WIDTH], float Cref[C_HEIGHT][C_WIDTH]);
void checkCUDAError(const char *msg);

__constant__ int BLOCK_SIZE;

__global__ void matrixMulCUDASharedMemory() {
    extern __shared__ float sm[];
    float *As = &sm[0];
    float *Bs = &sm[BLOCK_SIZE * BLOCK_SIZE];

    float Csub = 0;

    // iterate through the number of sub matrices of A and B
    const int a_y = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    const int b_x = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    for (int i = 0; i < NUM_SUBS; ++i) {
        //TODO: Calculate indices of A and B matrix required to load the shared block of memory
        const int a_x = i * BLOCK_SIZE + threadIdx.x;
        const int b_y = i * BLOCK_SIZE + threadIdx.y;

        //TODO: Each thread should load a single element of sub block of matrices A an B into shared memory
        As[threadIdx.y * BLOCK_SIZE + threadIdx.x] = d_A[a_y][a_x]; // global memory load with stride of 1, SM bank with stride of 1
        Bs[threadIdx.y * BLOCK_SIZE + threadIdx.x] = d_B[b_y][b_x]; // global memory load with stride of 1, SM bank with stride of 1

        // Sync to ensure sub matrix is fully loaded
        __syncthreads();

        //TODO: sum products of A and B sub matrices
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[threadIdx.y * BLOCK_SIZE + k] * Bs[k * BLOCK_SIZE + threadIdx.x]; // conflict free loads regardless of BLOCK_SIZE due to bradcast read and stride of 1
        }

        // Sync to prevent run ahead (blocks loading new SM values before others have completed)
        __syncthreads();
    }

    //TODO: caluclate the indices of sub matrix C
    const int c_x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int c_y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    // Store the product value of C matrix
    d_C[c_y][c_x] = Csub; // global memory load with stride of 1
}

int main(int argc, char **argv) {
    if (A_WIDTH != B_HEIGHT) {
        printf("Error: A_HEIGHT and B_WIDTH do not match\n");
    }

    const unsigned int mem_size_A = sizeof(float) * A_WIDTH * A_HEIGHT;
    const unsigned int mem_size_B = sizeof(float) * B_WIDTH * B_HEIGHT;
    const unsigned int mem_size_C = sizeof(float) * C_WIDTH * C_HEIGHT;

    // Initialise A
    for (unsigned int y = 0; y < A_HEIGHT; y++) {
        for (unsigned int x = 0; x < A_WIDTH; x++) {
            h_A[y][x] = (float)rand() / RAND_MAX;
        }
    }

    // Initialise B
    for (unsigned int y = 0; y < B_HEIGHT; y++) {
        for (unsigned int x = 0; x < B_WIDTH; x++) {
            h_B[y][x] = (float)rand() / RAND_MAX;
        }
    }

    // copy host memory to device
    cudaMemcpyToSymbol(d_A, h_A, mem_size_A);
    cudaMemcpyToSymbol(d_B, h_B, mem_size_B);
    checkCUDAError("CUDA memcpy");

    // Allocate CUDA events that we'll use for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    checkCUDAError("CUDA event creation");

    // Calculate block size
    int TPB, min_grid_size, block_size;
    cudaOccupancyMaxPotentialBlockSizeVariableSMem(&min_grid_size, &TPB, matrixMulCUDASharedMemory, requiredSM, 0);
    TPB = (int)pow(4, floor(log(TPB) / log(4))); // round to nearest square power 2
    block_size = (int)sqrt(TPB);
    cudaMemcpyToSymbol(BLOCK_SIZE, &block_size, sizeof(int));

    // Setup execution parameters
    dim3 grid(C_WIDTH / block_size, C_HEIGHT / block_size);
    dim3 threads(block_size, block_size);
    cudaEventRecord(start);
    matrixMulCUDASharedMemory << < grid, threads, requiredSM(block_size * block_size) >> > ();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    checkCUDAError("CUDA kernel execution and timing");

    float msec;
    cudaEventElapsedTime(&msec, start, stop);
    cudaDeviceSynchronize();
    checkCUDAError("CUDA timing");

    // Compute the ocupancy
    int maxActiveBlocks;
    cudaDeviceProp props;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, matrixMulCUDASharedMemory, block_size * block_size, 0);
    cudaGetDeviceProperties(&props, 0);
    const float occupancy = (maxActiveBlocks * block_size * block_size) / (float)props.maxThreadsPerMultiProcessor;

    // Copy result from device to host
    cudaMemcpyFromSymbol(h_C, d_C, mem_size_C);
    checkCUDAError("CUDA memcpy results");

    // Compute reference CPU version
    matrixMulCPU(h_A, h_B, h_C_ref);

    // Check for errors
    const unsigned int errors = matrixMulTest(h_C, h_C_ref);
    if (errors) {
        printf("%d total errors\n", errors);
    } else {
        printf("Test passed successfully\n");
    }

    printf("Kernel time was %f with block size %d and theoretical occupancy of %f\n", msec, block_size, occupancy);
}

int requiredSM(int TPB) {
    return 2 * TPB * sizeof(float);
}

void matrixMulCPU(float A[A_HEIGHT][A_WIDTH], float B[B_HEIGHT][B_WIDTH], float C[C_HEIGHT][C_WIDTH]) {
    for (int y = 0; y < C_HEIGHT; y++) {
        for (int x = 0; x < C_WIDTH; x++) {
            C[y][x] = 0;

            for (int k = 0; k < A_WIDTH; k++) {
                C[y][x] += A[y][k] * B[k][x];
            }
        }
    }
}

int matrixMulTest(float C[C_HEIGHT][C_WIDTH], float Cref[C_HEIGHT][C_WIDTH]) {
    int errors = 0;

    for (int y = 0; y < C_HEIGHT; y++) {
        for (int x = 0; x < C_WIDTH; x++) {
            // loss of accuracy due to CUDA's fused multiply add
            if (abs(C[y][x] - Cref[y][x]) > 0.001f) {
                errors++;
                printf("Device item c[%d][%d] = %f does not mach host result %f\n", y, x, C[y][x], Cref[y][x]);
            }
        }
    }

    return errors;
}

void checkCUDAError(const char *msg) {
    const cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
