#include <cstdio>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#include "NBody.h"
#include "NBodyVisualiser.h"
#include "nbody_data.h"

#define THREADS_PER_BLOCK 256

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
extern size_t nbodies_soa_size;
extern size_t activity_map_size;

// CUDA related variables
static dim3 nbodies_blocksPerGrid;
static dim3 activity_map_blocksPerGrid;

static nbody_soa h_nbodies, *d_nbodies;
static float2 *d_force_sum;
static float *d_activity_map;

// These pointers are required for (nbody_soa *) to work
float *d_x, *d_y, *d_vx, *d_vy, *d_m;

static __constant__ unsigned int c_N;
static __constant__ unsigned int c_D;
static __constant__ unsigned int c_grid_size;
static __constant__ float c_normalising_factor;

// Function declarations
static void step_gpu() noexcept;
static void allocate_memory() noexcept;
static void initialise_device() noexcept;
static void check(cudaError_t err, char const *func, int line) noexcept;

/**
 * Compute the summation of forces for each n-body
 *
 * @param d_nbodies   A device pointer to an n-bodies array (Array of Structures)
 * @param d_force_sum A device pointer to the force summation result of each n-body
 */
__global__ void compute_force(float2 *d_force_sum, nbody_soa *d_nbodies) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < c_N) {
        float2 local_sum = make_float2(0, 0);

        // Summation of forces for the current n-body
        for (unsigned int j = 0; j < c_N; ++j) {
            const float2 dist = make_float2(d_nbodies->x[j] - d_nbodies->x[i], d_nbodies->y[j] - d_nbodies->y[i]);
            const float inv_dist = rsqrtf(dist.x * dist.x + dist.y * dist.y + SOFTENING_SQUARE);
            const float m_div_mag = d_nbodies->m[j] * inv_dist * inv_dist * inv_dist;

            local_sum.x += m_div_mag * dist.x;
            local_sum.y += m_div_mag * dist.y;
        }

        d_force_sum[i] = local_sum;
    }
}

/**
 * Update the position and velocity of each n-body.
 * Also increase the count of each cell in activity map
 *
 * @param d_nbodies      A device pointer to an n-bodies array (Array of Structures)
 * @param d_force_sum    A device pointer to the force summation result of each n-body
 * @param d_activity_map A device pointer to an activity map array
 */
__global__ void update_body(nbody_soa *d_nbodies, float2 *d_force_sum, float *d_activity_map) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < c_N) {
        /* Movement */
        // Calculate position vector, do this first as it depends on current velocity
        d_nbodies->x[i] += dt * d_nbodies->vx[i];
        d_nbodies->y[i] += dt * d_nbodies->vy[i];

        // Calculate velocity vector, force and acceleration are computed together
        d_nbodies->vx[i] += dt_MUL_G * d_force_sum[i].x;
        d_nbodies->vy[i] += dt_MUL_G * d_force_sum[i].y;

        /* compute the position for a body in the `activity_map`
         * and increase the corresponding body count */
        const unsigned int col = static_cast<unsigned int>(d_nbodies->x[i] * static_cast<float>(c_D));
        const unsigned int row = static_cast<unsigned int>(d_nbodies->y[i] * static_cast<float>(c_D));

        // Do not update `activity_map` if n-body is out of grid area
        if (row < c_D && col < c_D) {
            atomicAdd(&d_activity_map[c_D * row + col], 1);
        }
    }
}

/**
 * Kernel function to normalise the activity map
 *
 * @param d_activity_map A device pointer to an activity map array
 */
__global__ void normalise_activity_map(float *d_activity_map) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Prevent reading `d_activity_map` beyond allocated (Unavoidable divergent branch)
    if (i < c_grid_size) {
        d_activity_map[i] *= c_normalising_factor;
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

    // Initialise host data and GPU device
    initialise_data_soa(&h_nbodies);
    initialise_device();

    if (I == 0) {
        initViewer(N, D, CUDA, &step_gpu);
        setNBodyPositions2f(d_x, d_y);
        setActivityMapData(d_activity_map);
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

    // Free device memory
    checkCudaError(cudaFree(d_x));
    checkCudaError(cudaFree(d_y));
    checkCudaError(cudaFree(d_vx));
    checkCudaError(cudaFree(d_vy));
    checkCudaError(cudaFree(d_m));
    checkCudaError(cudaFree(d_nbodies));
    checkCudaError(cudaFree(d_activity_map));
    checkCudaError(cudaFree(d_force_sum));
    checkCudaError(cudaDeviceReset());

    return 0;
}

/**
 * Perform the main simulation of the NBody system on the CPU
 */
static void step_gpu() noexcept {
    // Clear the activity map of previous step
    checkCudaError(cudaMemset(d_activity_map, 0, activity_map_size));

    compute_force << <nbodies_blocksPerGrid, THREADS_PER_BLOCK >> > (d_force_sum, d_nbodies);
    update_body << <nbodies_blocksPerGrid, THREADS_PER_BLOCK >> > (d_nbodies, d_force_sum, d_activity_map);
    normalise_activity_map << <activity_map_blocksPerGrid, THREADS_PER_BLOCK >> > (d_activity_map);
}

/**
 * Allocate required memory for host and device
 */
static void allocate_memory() noexcept {
    // Host memory
    h_nbodies.x = static_cast<float *>(malloc(nbodies_soa_size));
    h_nbodies.y = static_cast<float *>(malloc(nbodies_soa_size));
    h_nbodies.vx = static_cast<float *>(malloc(nbodies_soa_size));
    h_nbodies.vy = static_cast<float *>(malloc(nbodies_soa_size));
    h_nbodies.m = static_cast<float *>(malloc(nbodies_soa_size));

    // Device memory
    checkCudaError(cudaMalloc(reinterpret_cast<void **>(&d_nbodies), sizeof(nbody_soa)));
    checkCudaError(cudaMalloc(reinterpret_cast<void **>(&d_x), nbodies_soa_size));
    checkCudaError(cudaMalloc(reinterpret_cast<void **>(&d_y), nbodies_soa_size));
    checkCudaError(cudaMalloc(reinterpret_cast<void **>(&d_vx), nbodies_soa_size));
    checkCudaError(cudaMalloc(reinterpret_cast<void **>(&d_vy), nbodies_soa_size));
    checkCudaError(cudaMalloc(reinterpret_cast<void **>(&d_m), nbodies_soa_size));
    checkCudaError(cudaMalloc(reinterpret_cast<void **>(&d_activity_map), activity_map_size));
    checkCudaError(cudaMalloc(reinterpret_cast<void **>(&d_force_sum), sizeof(float2) * N));
}

/**
 * Initialise the device with required values
 */
static void initialise_device() noexcept {
    // Copy host data to individual pointers first
    checkCudaError(cudaMemcpy(d_x, h_nbodies.x, nbodies_soa_size, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_y, h_nbodies.y, nbodies_soa_size, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_vx, h_nbodies.vx, nbodies_soa_size, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_vy, h_nbodies.vy, nbodies_soa_size, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_m, h_nbodies.m, nbodies_soa_size, cudaMemcpyHostToDevice));

    // Copy individual pointers to SoA pointer
    checkCudaError(cudaMemcpy(d_nbodies, &h_nbodies, sizeof(nbody_soa), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(&d_nbodies->x, &d_x, sizeof(float *), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(&d_nbodies->y, &d_y, sizeof(float *), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(&d_nbodies->vx, &d_vx, sizeof(float *), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(&d_nbodies->vy, &d_vy, sizeof(float *), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(&d_nbodies->m, &d_m, sizeof(float *), cudaMemcpyHostToDevice));

    // Copy runtime constants into constant memory
    cudaMemcpyToSymbol(c_N, &N, sizeof N);
    cudaMemcpyToSymbol(c_D, &D, sizeof D);
    cudaMemcpyToSymbol(c_grid_size, &grid_size, sizeof grid_size);
    cudaMemcpyToSymbol(c_normalising_factor, &normalising_factor, sizeof normalising_factor);

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
static void check(cudaError_t err, char const *func, const int line) noexcept {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error \"%s\" at line %d: %s\n", cudaGetErrorString(err), line, func);
        exit(EXIT_FAILURE);
    }
}
