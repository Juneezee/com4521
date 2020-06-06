#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <ctime>

#ifndef __CUDACC__
#define __CUDACC__
#endif

// CUDA runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

//thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/random.h>
#include <thrust/generate.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#define TPB 256

#define NUM_PARTICLES 16384
#define ENV_DIM 32.0f
#define INTERACTION_RANGE 4.0f
#define ENV_BIN_DIM ((unsigned int)(ENV_DIM/INTERACTION_RANGE))
#define ENV_BINS (ENV_BIN_DIM*ENV_BIN_DIM)

typedef struct key_values {
    int sorting_key[NUM_PARTICLES];
    int value[NUM_PARTICLES];
} key_values;

typedef struct particles {
    float2 location[NUM_PARTICLES];
    int nn_key[NUM_PARTICLES];
} particles;

typedef struct environment {
    int count[ENV_BINS];
    int start_index[ENV_BINS];
} environment;

__global__ void particleNNSearch(particles *p, environment *env);
__global__ void keyValues(particles *p, key_values *kv);
__global__ void reorderParticles(key_values *kv, particles *p, particles *p_sorted);
__global__ void histogramParticles(particles *p, environment *env);
__device__ __host__ int2 binLocation(float2 location);
__device__ __host__ int binIndex(int2 bin);

void particlesCPU();
void particlesGPU();
void initParticles(particles *p);
int checkResults(const char *name, particles *p);
void keyValuesCPU(particles *p, key_values *kv);
void sortKeyValuesCPU(key_values *kv);
void reorderParticlesCPU(key_values *kv, particles *p, particles *p_sorted);
void histogramParticlesCPU(particles *p, environment *env);
void prefixSumEnvironmentCPU(environment *env);

void checkCUDAError(const char *msg);

/* GPU Kernels */

__global__ void particleNNSearch(particles *p, environment *env) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    //get location
    const float2 location = p->location[idx];
    const int2 bin = binLocation(location);
    int nn = -1;

    //check all neighbouring bins of particle (9 in total) - no boundary wrapping
    float dist_sq = ENV_DIM * ENV_DIM; //a big number

    for (int x = bin.x - 1; x <= bin.x + 1; x++) {
        //no wrapping
        if (x < 0 || x >= ENV_BIN_DIM)
            continue;

        for (int y = bin.y - 1; y <= bin.y + 1; y++) {
            //no wrapping
            if (y < 0 || y >= ENV_BIN_DIM)
                continue;

            //get the bin index
            const int bin_index = binIndex(make_int2(x, y));

            //get start index of the bin
            const int bin_start_index = env->start_index[bin_index];

            //get the count of the bin
            const int bin_count = env->count[bin_index];

            //loop through particles to find nearest neighbour
            for (int i = bin_start_index; i < bin_start_index + bin_count; i++) {
                const float2 n_location = p->location[i];
                if (i != idx) {
                    //cant be closest to itself
                    //distance check (no need to square root)
                    const float n_dist_sq = (n_location.x - location.x) * (n_location.x - location.x) + (n_location.y - location.y) * (n_location.y - location.y);
                    if (n_dist_sq < dist_sq) {
                        //we have found a new nearest neighbour if it is within the range
                        if (n_dist_sq < INTERACTION_RANGE * INTERACTION_RANGE) {
                            dist_sq = n_dist_sq;
                            nn = i;
                        }
                    }
                }
            }
        }
    }

    //write nearest neighbour
    p->nn_key[idx] = nn;
}

/* Thrust Implementation Additional Kernels */

__global__ void keyValues(particles *p, key_values *kv) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    kv->value[idx] = idx;
    kv->sorting_key[idx] = binIndex(binLocation(p->location[idx]));
}

__global__ void reorderParticles(key_values *kv, particles *p, particles *p_sorted) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int old_index = kv->value[idx];
    p_sorted->location[idx] = p->location[old_index];
}

__global__ void histogramParticles(particles *p, environment *env) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int bin_location = binIndex(binLocation(p->location[idx])); //recalculate the sort value
    atomicAdd(&env->count[bin_location], 1);
}

__device__ __host__ int2 binLocation(float2 location) {
    int bin_x = static_cast<int>(location.x / INTERACTION_RANGE);
    int bin_y = static_cast<int>(location.y / INTERACTION_RANGE);
    return make_int2(bin_x, bin_y);
}

__device__ __host__ int binIndex(int2 bin) {
    return bin.x + bin.y * ENV_BIN_DIM;
}

/* Host Functions*/

int main(int argc, char **argv) {
    particlesCPU();
    particlesGPU();

    return 0;
}

void particlesCPU() {
    environment *h_env;
    environment *d_env;
    particles *h_particles;
    particles *h_particles_sorted;
    particles *d_particles;
    particles *d_particles_sorted;
    key_values *h_key_values;
    key_values *d_key_values;

    float time;
    clock_t begin, end;
    int errors;

    //allocate host memory (pinned)
    h_env = static_cast<environment *>(malloc(sizeof(environment)));
    h_particles = static_cast<particles *>(malloc(sizeof(particles)));
    h_particles_sorted = static_cast<particles *>(malloc(sizeof(particles)));
    h_key_values = static_cast<key_values *>(malloc(sizeof(key_values)));
    checkCUDAError("CPU version: Host malloc");

    //allocate device memory
    cudaMalloc((void **)&d_env, sizeof(environment));
    cudaMalloc((void **)&d_particles, sizeof(particles));
    cudaMalloc((void **)&d_particles_sorted, sizeof(particles));
    cudaMalloc((void **)&d_key_values, sizeof(key_values));
    checkCUDAError("CPU version: Device malloc");

    //set host data to 0
    memset(h_env, 0, sizeof(environment));
    memset(h_particles, 0, sizeof(particles));
    memset(h_key_values, 0, sizeof(key_values));

    //set device data to 0
    cudaMemset(d_env, 0, sizeof(environment));
    cudaMemset(d_particles, 0, sizeof(particles));
    cudaMemset(d_key_values, 0, sizeof(key_values));
    checkCUDAError("CPU version: Device memset");

    //init some particle data
    initParticles(h_particles);

    /* CPU implementation */
    cudaDeviceSynchronize();
    begin = clock();

    //key value pairs
    keyValuesCPU(h_particles, h_key_values);
    //sort particles on CPU
    sortKeyValuesCPU(h_key_values);
    //reorder particles
    reorderParticlesCPU(h_key_values, h_particles, h_particles_sorted);
    //histogram particle counts
    histogramParticlesCPU(h_particles_sorted, h_env);
    //prefix sum the environment bin locations
    prefixSumEnvironmentCPU(h_env);
    //host to device copy
    cudaMemcpy(d_particles_sorted, h_particles_sorted, sizeof(particles), cudaMemcpyHostToDevice);
    cudaMemcpy(d_env, h_env, sizeof(environment), cudaMemcpyHostToDevice);
    checkCUDAError("CPU version: Host 2 Device");
    //particle nearest neighbour kernel
    particleNNSearch << <NUM_PARTICLES / TPB, TPB >> > (d_particles_sorted, d_env);
    checkCUDAError("CPU version: CPU version Kernel");
    //device to host copy
    cudaMemcpy(h_particles_sorted, d_particles_sorted, sizeof(particles), cudaMemcpyDeviceToHost);
    checkCUDAError("CPU version: Device 2 Host");

    //calculate timing
    cudaDeviceSynchronize();
    end = clock();
    time = static_cast<float>(end - begin) / CLOCKS_PER_SEC;

    errors = checkResults("CPU", h_particles_sorted);
    printf("CPU NN Search completed in %f seconds with %d errors\n", time, errors);

    //free host and device memory
    free(h_env);
    free(h_particles);
    free(h_particles_sorted);
    free(h_key_values);
    cudaFree(d_env);
    cudaFree(d_particles);
    cudaFree(d_particles_sorted);
    cudaFree(d_key_values);
    checkCUDAError("CPU version: CUDA free");
}

void particlesGPU() {
    environment *d_env;
    particles *d_particles;
    particles *d_particles_sorted;
    key_values *d_key_values;

    //allocate host memory (pinned)
    environment *h_env = static_cast<environment *>(malloc(sizeof(environment)));
    particles *h_particles = static_cast<particles *>(malloc(sizeof(particles)));
    particles *h_particles_sorted = static_cast<particles *>(malloc(sizeof(particles)));
    key_values *h_key_values = static_cast<key_values *>(malloc(sizeof(key_values)));
    checkCUDAError("GPU version: Host malloc");

    //allocate device memory
    cudaMalloc((void **)&d_env, sizeof(environment));
    cudaMalloc((void **)&d_particles, sizeof(particles));
    cudaMalloc((void **)&d_particles_sorted, sizeof(particles));
    cudaMalloc((void **)&d_key_values, sizeof(key_values));
    checkCUDAError("GPU version: Device malloc");

    //set host data to 0
    memset(h_env, 0, sizeof(environment));
    memset(h_particles, 0, sizeof(particles));
    memset(h_key_values, 0, sizeof(key_values));

    //set device data to 0
    cudaMemset(d_env, 0, sizeof(environment));
    cudaMemset(d_particles, 0, sizeof(particles));
    cudaMemset(d_key_values, 0, sizeof(key_values));
    checkCUDAError("GPU version: Device memset");
    //init some particle data
    initParticles(h_particles);

    /* Thrust Implementation */
    cudaDeviceSynchronize();
    const clock_t begin = clock();

    //Exercise 1.1) Copy from host to device
    cudaMemcpy(d_particles, h_particles, sizeof(particles), cudaMemcpyHostToDevice);
    checkCUDAError("GPU version: Host 2 Device");

    //Exercise 1.2) generate key value pairs on device
    keyValues << <NUM_PARTICLES / TPB, TPB >> > (d_particles, d_key_values);
    checkCUDAError("GPU version: Device keyValues");

    //Exercise 1.3) sort by key
    sort_by_key(thrust::device_ptr<int>(d_key_values->sorting_key), thrust::device_ptr<int>(d_key_values->sorting_key + NUM_PARTICLES), thrust::device_ptr<int>(d_key_values->value));
    checkCUDAError("GPU version: Thrust sort");

    //Exercise 1.4) re-order
    reorderParticles << <NUM_PARTICLES / TPB, TPB >> > (d_key_values, d_particles, d_particles_sorted);
    checkCUDAError("GPU version: Device reorder");

    //Exercise 1.5) histogram
    histogramParticles << <NUM_PARTICLES / TPB, TPB >> > (d_particles_sorted, d_env);
    checkCUDAError("GPU version: Device Histogram");

    //Exercise 1.6) thrust prefix sum
    exclusive_scan(thrust::device_pointer_cast(d_env->count), thrust::device_pointer_cast(d_env->count + ENV_BINS), thrust::device_pointer_cast(d_env->start_index));
    checkCUDAError("GPU version: Thrust scan");

    //particle nearest neighbour kernel
    particleNNSearch << <NUM_PARTICLES / TPB, TPB >> > (d_particles_sorted, d_env);
    checkCUDAError("GPU version: Kernel");

    //device to host copy
    cudaMemcpy(h_particles_sorted, d_particles_sorted, sizeof(particles), cudaMemcpyDeviceToHost);
    checkCUDAError("GPU version: Device 2 Host");

    //calculate timing
    cudaDeviceSynchronize();
    const clock_t end = clock();
    const float time = static_cast<float>(end - begin) / CLOCKS_PER_SEC;

    /* print results and clean up*/
    const int errors = checkResults("GPU", h_particles_sorted);
    printf("GPU NN Search completed in %f seconds with %d errors\n", time, errors);

    //free host and device memory
    free(h_env);
    free(h_particles);
    free(h_particles_sorted);
    free(h_key_values);
    cudaFree(d_env);
    cudaFree(d_particles);
    cudaFree(d_particles_sorted);
    cudaFree(d_key_values);
    checkCUDAError("GPU version: CUDA free");
}

void initParticles(particles *p) {
    // seed
    srand(123);

    // random positions
    for (float2 &i : p->location) {
        const float rand_x = rand() / static_cast<float>(RAND_MAX) * ENV_DIM;
        const float rand_y = rand() / static_cast<float>(RAND_MAX) * ENV_DIM;
        const float2 location = make_float2(rand_x, rand_y);
        i = location;
    }
}

int checkResults(const char *name, particles *p) {
    int errors = 0;

    for (int i = 0; i < NUM_PARTICLES; i++) {
        const float2 location = p->location[i];
        float dist_sq = ENV_DIM * ENV_DIM; //a big number
        int cpu_nn = -1;

        // find nearest neighbour on CPU
        for (int j = 0; j < NUM_PARTICLES; j++) {
            const float2 n_location = p->location[j];
            if (j != i) {
                // cant be closest to itself
                // distance check (no need to square root)
                const float n_dist_sq = (n_location.x - location.x) * (n_location.x - location.x) + (n_location.y - location.y) * (n_location.y - location.y);
                if (n_dist_sq < dist_sq) {
                    // we have found a new nearest neighbour if it is within the range
                    if (n_dist_sq < INTERACTION_RANGE * INTERACTION_RANGE) {
                        dist_sq = n_dist_sq;
                        cpu_nn = j;
                    }
                }
            }
        }

        if (p->nn_key[i] != cpu_nn) {
            fprintf(stderr, "Error: %s NN for index %d is %d, Ref NN is %u\n", name, i, p->nn_key[i], cpu_nn);
            errors++;
        }
    }

    return errors;
}

void keyValuesCPU(particles *p, key_values *kv) {
    // Random positions
    for (int i = 0; i < NUM_PARTICLES; i++) {
        const float2 location = p->location[i];
        kv->value[i] = i;
        kv->sorting_key[i] = binIndex(binLocation(location));
    }
}

/**
 * Simple (inefficient) CPU bubble sort
 */
void sortKeyValuesCPU(key_values *kv) {
    for (int i = 0; i < NUM_PARTICLES - 1; i++) {
        for (int j = 0; j < NUM_PARTICLES - i - 1; j++) {
            if (kv->sorting_key[j] > kv->sorting_key[j + 1]) {
                // Swap values
                const int swap_key = kv->value[j];
                const int swap_sort_value = kv->sorting_key[j];

                kv->value[j] = kv->value[j + 1];
                kv->sorting_key[j] = kv->sorting_key[j + 1];

                kv->value[j + 1] = swap_key;
                kv->sorting_key[j + 1] = swap_sort_value;
            }
        }
    }
}

/**
 * Re-order based on the old key
 */
void reorderParticlesCPU(key_values *kv, particles *p, particles *p_sorted) {
    for (int i = 0; i < NUM_PARTICLES; i++) {
        const int old_index = kv->value[i];
        p_sorted->location[i] = p->location[old_index];
    }
}

/**
 * Loop through particles and increase the bin count for each environment bin
 */
void histogramParticlesCPU(particles *p, environment *env) {
    for (int i = 0; i < NUM_PARTICLES - 1; i++) {
        const int bin_location = binIndex(binLocation(p->location[i])); //recalculate the sort value
        env->count[bin_location]++;
    }
}

/**
 * Serial prefix sum
 */
void prefixSumEnvironmentCPU(environment *env) {
    int sum = 0;

    for (int i = 0; i < ENV_BINS; i++) {
        env->start_index[i] = sum;
        sum += env->count[i];
    }
}

void checkCUDAError(const char *msg) {
    const cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
