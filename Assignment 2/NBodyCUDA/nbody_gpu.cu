#include <cstdio>
#include <cmath>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "NBody.h"
#include "NBodyVisualiser.h"
#include "nbody_data.h"

#define THREADS_PER_BLOCK 32

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
extern size_t nbodies_size;
extern size_t activity_map_size;

static nbody *h_nbodies, *d_nbodies;
static float *h_activity_map, *d_activity_map;

dim3 nbodies_blocksPerGrid;
dim3 activity_map_blocksPerGrid;

// Function declarations
static void step_gpu() noexcept;
static void allocate_memory() noexcept;
static void check(cudaError_t err, char const *func, int line) noexcept;

__global__ void parallelise_each_body(nbody *d_nbodies, float *d_activity_map, const unsigned int N, const unsigned int D) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        /* Force */
        float sum_x = 0, sum_y = 0;

        for (unsigned int j = 0; j < N; ++j) {
            const float dist_x = d_nbodies[j].x - d_nbodies[i].x;
            const float dist_y = d_nbodies[j].y - d_nbodies[i].y;
            const float mag_add_soft = dist_x * dist_x + dist_y * dist_y + SOFTENING_SQUARE;
            const float m_div_soft = d_nbodies[j].m / (mag_add_soft * sqrtf(mag_add_soft));

            sum_x += m_div_soft * dist_x;
            sum_y += m_div_soft * dist_y;
        }

        /* Movement */
        // Calculate position vector, do this first as it depends on current velocity
        d_nbodies[i].x += dt * d_nbodies[i].vx;
        d_nbodies[i].y += dt * d_nbodies[i].vy;

        // Calculate velocity vector, force and acceleration are computed together
        d_nbodies[i].vx += dt_MUL_G * sum_x;
        d_nbodies[i].vy += dt_MUL_G * sum_y;

        /* compute the position for a body in the `activity_map`
         * and increase the corresponding body count */
        const unsigned int col = static_cast<unsigned int>(d_nbodies[i].x * static_cast<float>(D));
        const unsigned int row = static_cast<unsigned int>(d_nbodies[i].y * static_cast<float>(D));

        // Do not update `activity_map` if n-body is out of grid area
        if (row < D && col < D) {
            atomicAdd(&d_activity_map[D * row + col], 1);
        }
    }
}

__global__ void normalise_activity_map(float *d_activity_map, const unsigned int grid_size, const float normalising_factor) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < grid_size) {
        d_activity_map[i] *= normalising_factor;
    }
}

int main_gpu() noexcept {
    // Allocate any host memory and device memory
    allocate_memory();

    // Initialise host data
    initialise_data_aos(h_nbodies);

    // Copy host data to device
    checkCudaError(cudaMemcpy(d_nbodies, h_nbodies, nbodies_size, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_activity_map, h_activity_map, activity_map_size, cudaMemcpyHostToDevice));

    // Calculate the required blocks
    nbodies_blocksPerGrid = { (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1, 1 };
    activity_map_blocksPerGrid = { (grid_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1, 1 };

    if (I == 0) {
        initViewer(N, D, CUDA, &step_gpu);
        setNBodyPositions(d_nbodies);
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
    free(h_nbodies);
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
    h_nbodies = static_cast<nbody *>(malloc(nbodies_size));
    if (h_nbodies == nullptr) {
        fprintf(stderr, "error: failed to allocate memory: h_nbodies\n");
        exit(EXIT_FAILURE);
    }

    h_activity_map = static_cast<float *>(malloc(activity_map_size));
    if (h_activity_map == nullptr) {
        fprintf(stderr, "error: failed to allocate memory: h_activity_map");
        exit(EXIT_FAILURE);
    }

    // Device memory
    checkCudaError(cudaMalloc((void **)&d_nbodies, nbodies_size));
    checkCudaError(cudaMalloc((void **)&d_activity_map, activity_map_size));
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
