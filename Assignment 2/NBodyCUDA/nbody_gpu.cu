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

static nbody_soa h_nbodies, d_nbodies;
static float2 *d_force_sum;
static float *d_activity_map;

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
 * @param x           A device pointer to a float array of length N containing the x position of bodies
 * @param y           A device pointer to a float array of length N containing the y position of bodies
 * @param m           A device pointer to a float array of length N containing the mass of bodies
 * @param d_force_sum A device pointer to the force summation result of each n-body
 */
__global__ void compute_force(const float *__restrict__ x,
                              const float *__restrict__ y,
                              const float *__restrict__ m,
                              float2 *__restrict__ d_force_sum) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float3 s_nbodies[THREADS_PER_BLOCK];

    float2 local_sum = make_float2(0, 0);

    // Divide N bodies into sub-blocks
    for (unsigned int sub_block = 0; sub_block < gridDim.x; ++sub_block) {
        const unsigned int sub_i = sub_block * THREADS_PER_BLOCK + threadIdx.x;

        // Load data into shared memory. `sub_i < c_N` to avoid reading beyond allocated
        s_nbodies[threadIdx.x] = sub_i < c_N
                                     ? make_float3(x[sub_i], y[sub_i], m[sub_i])
                                     : make_float3(0, 0, 0);
        __syncthreads();

        // Summation of forces for the current n-body
        if (i < c_N) {
            for (float3 &s_nbody : s_nbodies) {
                const float2 dist = make_float2(s_nbody.x - x[i], s_nbody.y - y[i]);
                const float inv_dist = rsqrtf(dist.x * dist.x + dist.y * dist.y + SOFTENING_SQUARE);
                const float m_div_mag = s_nbody.z * inv_dist * inv_dist * inv_dist;

                local_sum.x += m_div_mag * dist.x;
                local_sum.y += m_div_mag * dist.y;
            }
        }

        __syncthreads();
    }

    // Store the result to global memory
    if (i < c_N) {
        d_force_sum[i] = local_sum;
    }
}

/**
 * Update the position and velocity of each n-body.
 * Also increase the count of each cell in activity map
 *
 * @param x              A device pointer to a float array of length N containing the x position of bodies
 * @param y              A device pointer to a float array of length N containing the y position of bodies
 * @param vx             A device pointer to a float array of length N containing the x velocity of bodies
 * @param vy             A device pointer to a float array of length N containing the y velocity of bodies
 * @param d_force_sum    A device pointer to the force summation result of each n-body
 * @param d_activity_map A device pointer to an activity map array
 */
__global__ void update_body(float *__restrict__ x,
                            float *__restrict__ y,
                            float *__restrict__ vx,
                            float *__restrict__ vy,
                            const float2 *__restrict__ d_force_sum,
                            float *__restrict__ d_activity_map) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < c_N) {
        /* Movement */
        // Calculate position vector, do this first as it depends on current velocity
        x[i] += dt * vx[i];
        y[i] += dt * vy[i];

        // Calculate velocity vector, force and acceleration are computed together
        vx[i] += dt_MUL_G * d_force_sum[i].x;
        vy[i] += dt_MUL_G * d_force_sum[i].y;

        /* compute the position for a body in the `activity_map`
         * and increase the corresponding body count */
        const unsigned int col = static_cast<unsigned int>(x[i] * static_cast<float>(c_D));
        const unsigned int row = static_cast<unsigned int>(y[i] * static_cast<float>(c_D));

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
__global__ void normalise_activity_map(float *__restrict__ d_activity_map) {
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
        setNBodyPositions2f(d_nbodies.x, d_nbodies.y);
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
    checkCudaError(cudaFree(d_nbodies.x));
    checkCudaError(cudaFree(d_nbodies.y));
    checkCudaError(cudaFree(d_nbodies.vx));
    checkCudaError(cudaFree(d_nbodies.vy));
    checkCudaError(cudaFree(d_nbodies.m));
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

    compute_force << <nbodies_blocksPerGrid, THREADS_PER_BLOCK >> >(d_nbodies.x,
                                                                    d_nbodies.y,
                                                                    d_nbodies.m,
                                                                    d_force_sum);
    update_body << <nbodies_blocksPerGrid, THREADS_PER_BLOCK >> >(d_nbodies.x,
                                                                  d_nbodies.y,
                                                                  d_nbodies.vx,
                                                                  d_nbodies.vy,
                                                                  d_force_sum,
                                                                  d_activity_map);
    normalise_activity_map << <activity_map_blocksPerGrid, THREADS_PER_BLOCK >> >(d_activity_map);
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
    checkCudaError(cudaMalloc(reinterpret_cast<void **>(&d_nbodies.x), nbodies_soa_size));
    checkCudaError(cudaMalloc(reinterpret_cast<void **>(&d_nbodies.y), nbodies_soa_size));
    checkCudaError(cudaMalloc(reinterpret_cast<void **>(&d_nbodies.vx), nbodies_soa_size));
    checkCudaError(cudaMalloc(reinterpret_cast<void **>(&d_nbodies.vy), nbodies_soa_size));
    checkCudaError(cudaMalloc(reinterpret_cast<void **>(&d_nbodies.m), nbodies_soa_size));
    checkCudaError(cudaMalloc(reinterpret_cast<void **>(&d_activity_map), activity_map_size));
    checkCudaError(cudaMalloc(reinterpret_cast<void **>(&d_force_sum), sizeof(float2) * N));
}

/**
 * Initialise the device with required values
 */
static void initialise_device() noexcept {
    // Copy host data to device
    checkCudaError(cudaMemcpy(d_nbodies.x, h_nbodies.x, nbodies_soa_size, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_nbodies.y, h_nbodies.y, nbodies_soa_size, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_nbodies.vx, h_nbodies.vx, nbodies_soa_size, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_nbodies.vy, h_nbodies.vy, nbodies_soa_size, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_nbodies.m, h_nbodies.m, nbodies_soa_size, cudaMemcpyHostToDevice));

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
