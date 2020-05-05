#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "NBody.h"
#include "NBodyVisualiser.h"
#include "nbody_data.h"

#define THREADS_PER_BLOCK 64

// This will output the proper CUDA error strings in the event
// that a CUDA host call returns an error
#define checkCudaError(val) check((val), #val, __LINE__)

// External variables defined in `nbody_args`
extern unsigned int N;
extern unsigned int D;
extern unsigned int I;

// External variables defined in `NBody`
extern unsigned int grid_size;
extern float normalising_factor;
extern size_t activity_map_size;

// CUDA related variables
static nbody_soa h_nbodies, d_nbodies;
static float *h_activity_map, *d_activity_map;

static dim3 nbodies_blocksPerGrid;
static dim3 activity_map_blocksPerGrid;

// Function declarations
static void step_gpu() noexcept;
static void allocate_memory() noexcept;
static void initialise_device() noexcept;
static void check(cudaError_t err, char const *func, int line) noexcept;

/**
 * Kernel function to parallelise each body (N threads)
 *
 * @param d_nbodies      A Structure of Arrays (SoA) layout of n-bodies
 * @param d_activity_map A device pointer to an activity map array
 * @param N              Number of bodies
 * @param D              Grid dimension
 */
__global__ void parallelise_each_body(nbody_soa d_nbodies, float *d_activity_map, const unsigned int N, const unsigned int D) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread represents a body. `i < N` to avoid reading beyond allocated.
    float4 body = i < N
        ? make_float4(d_nbodies.x[i], d_nbodies.y[i], d_nbodies.vx[i], d_nbodies.vy[i])
        : make_float4(0, 0, 0, 0);

    __shared__ float3 s_nbodies[THREADS_PER_BLOCK];

    /* Force */
    float2 sum = make_float2(0, 0);

    // Shared memory - divide d_nbodies into sub-blocks
    for (unsigned int sub_block = 0; sub_block < gridDim.x; ++sub_block) {
        const unsigned int sub_i = sub_block * THREADS_PER_BLOCK + threadIdx.x;

        // Load into shared memory. `sub_i < N` to avoid reading beyond allocated
        s_nbodies[threadIdx.x] = sub_i < N
            ? make_float3(d_nbodies.x[sub_i], d_nbodies.y[sub_i], d_nbodies.m[sub_i])
            : make_float3(0, 0, 0);
        __syncthreads();

        // Iterating the sub-block. `sub_i < N` to avoid calculating "leftover" threads
        for (unsigned int j = 0; sub_i < N && j < THREADS_PER_BLOCK; ++j) {
            const float2 dist = make_float2(s_nbodies[j].x - body.x, s_nbodies[j].y - body.y);
            const float inv_dist = rsqrtf(dist.x * dist.x + dist.y * dist.y + SOFTENING_SQUARE);
            const float m_div_mag = s_nbodies[j].z * inv_dist * inv_dist * inv_dist;

            sum.x += m_div_mag * dist.x;
            sum.y += m_div_mag * dist.y;
        }

        __syncthreads();
    }

    // Prevent reading `d_nbodies` beyond allocated (Unavoidable divergent branch)
    if (i < N) {
        /* Movement */
        // Calculate position vector, do this first as it depends on current velocity
        d_nbodies.x[i] = body.x + dt * body.z;
        d_nbodies.y[i] = body.y + dt * body.w;

        // Calculate velocity vector
        d_nbodies.vx[i] = body.z + dt_MUL_G * sum.x;
        d_nbodies.vy[i] = body.w + dt_MUL_G * sum.y;

        /* compute the position for `body` in the `activity_map`
         * and increase the corresponding body count */
        const unsigned int col = static_cast<unsigned int>(d_nbodies.x[i] * static_cast<float>(D));
        const unsigned int row = static_cast<unsigned int>(d_nbodies.y[i] * static_cast<float>(D));

        // Do not update `activity_map` if n-body is out of grid area
        if (row < D && col < D) {
            atomicAdd(&d_activity_map[D * row + col], 1);
        }
    }
}

/**
 * Kernel function to normalise the activity map
 *
 * @param d_activity_map     The activity map array stored in device.
 * @param grid_size          D * D. Runtime constant calculated in `NBody.cu`.
 * @param normalising_factor D / N. Runtime constant calculated in `NBody.cu`.
 */
__global__ void normalise_activity_map(float *d_activity_map, const unsigned int grid_size, const float normalising_factor) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Prevent reading `d_activity_map` beyond allocated (Unavoidable divergent branch)
    if (i < grid_size) {
        d_activity_map[i] *= normalising_factor;
    }
}

/**
 * Entry point of program for CUDA mode
 *
 * @return int The program exit code
 */
int main_gpu() noexcept {
    // Allocate any host memory and device memory
    allocate_memory();

    // Initialise host data
    initialise_data_soa(&h_nbodies);
    initialise_device();

    // Calculate the required blocks
    nbodies_blocksPerGrid = { (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1, 1 };
    activity_map_blocksPerGrid = { (grid_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1, 1 };

    if (I == 0) {
        initViewer(N, D, CUDA, &step_gpu);
        setNBodyPositions2f(d_nbodies.x, d_nbodies.y);
        setHistogramData(d_activity_map);
        startVisualisationLoop();
    } else {
        // CUDA event creation
        cudaEvent_t start, stop;
        checkCudaError(cudaEventCreate(&start));
        checkCudaError(cudaEventCreate(&stop));

        // Record timing of step_gpu()
        checkCudaError(cudaEventRecord(start));
        for (unsigned int i = 0; i < I; ++i) {
            step_gpu();
        }
        checkCudaError(cudaEventRecord(stop));
        checkCudaError(cudaEventSynchronize(stop));

        // Output total time taken
        float ms;
        checkCudaError(cudaEventElapsedTime(&ms, start, stop));

        const int seconds = static_cast<int>(ms) / 1000;
        printf("Execution time was %d seconds %d milliseconds\n", seconds, static_cast<int>(ms) % 1000);
    }

    // Free host memory
    free(h_nbodies.x);
    free(h_nbodies.y);
    free(h_nbodies.vx);
    free(h_nbodies.vy);
    free(h_nbodies.m);
    free(h_activity_map);

    // Free device memory
    checkCudaError(cudaDeviceReset());

    return 0;
}

/**
 * Perform the main simulation of the NBody system on the CPU
 */
static void step_gpu() noexcept {
    // Clear the activity map of previous step
    checkCudaError(cudaMemset(d_activity_map, 0, activity_map_size));

    parallelise_each_body << <nbodies_blocksPerGrid, THREADS_PER_BLOCK >> > (d_nbodies, d_activity_map, N, D);
    normalise_activity_map << <activity_map_blocksPerGrid, THREADS_PER_BLOCK >> > (d_activity_map, grid_size, normalising_factor);
}

/**
 * Allocate required memory for host and device
 */
static void allocate_memory() noexcept {
    // Host memory
    const size_t size = sizeof(float) * N;
    h_nbodies.x = static_cast<float *>(malloc(size));
    h_nbodies.y = static_cast<float *>(malloc(size));
    h_nbodies.vx = static_cast<float *>(malloc(size));
    h_nbodies.vy = static_cast<float *>(malloc(size));
    h_nbodies.m = static_cast<float *>(malloc(size));

    h_activity_map = static_cast<float *>(malloc(activity_map_size));
    if (h_activity_map == nullptr) {
        fprintf(stderr, "error: failed to allocate memory: h_activity_map");
        exit(EXIT_FAILURE);
    }

    // Device memory
    checkCudaError(cudaMalloc(reinterpret_cast<void **>(&d_nbodies.x), size));
    checkCudaError(cudaMalloc(reinterpret_cast<void **>(&d_nbodies.y), size));
    checkCudaError(cudaMalloc(reinterpret_cast<void **>(&d_nbodies.vx), size));
    checkCudaError(cudaMalloc(reinterpret_cast<void **>(&d_nbodies.vy), size));
    checkCudaError(cudaMalloc(reinterpret_cast<void **>(&d_nbodies.m), size));
    checkCudaError(cudaMalloc(reinterpret_cast<void **>(&d_activity_map), activity_map_size));
}

/**
 * Initialise the device with required values
 */
static void initialise_device() noexcept {
    // Copy host data to device
    const size_t size = sizeof(float) * N;
    checkCudaError(cudaMemcpy(d_nbodies.x, h_nbodies.x, size, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_nbodies.y, h_nbodies.y, size, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_nbodies.vx, h_nbodies.vx, size, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_nbodies.vy, h_nbodies.vy, size, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_nbodies.m, h_nbodies.m, size, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_activity_map, h_activity_map, activity_map_size, cudaMemcpyHostToDevice));

    // Calculate the required blocks
    nbodies_blocksPerGrid = { (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1, 1 };
    activity_map_blocksPerGrid = { (grid_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1, 1 };
}

/**
 * --- Used by checkCudaError macro ---
 * Check for CUDA error. Exit with failure if an error occurred.
 *
 * @param err The cudaError_t value
 * @param func The line of code that generated the error
 * @param line The line number
 */
static void check(cudaError_t err, char const *func, int line) noexcept {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error \"%s\" at line %d: %s\n", cudaGetErrorString(err), line, func);
        exit(EXIT_FAILURE);
    }
}
