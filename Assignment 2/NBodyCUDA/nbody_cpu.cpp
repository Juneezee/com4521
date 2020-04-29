#include <cstdio>
#include <cmath>
#include <omp.h>

#include "NBody.h"
#include "NBodyVisualiser.h"
#include "nbody_data.h"

// External variables defined in `nbody_args`
extern unsigned int N;
extern unsigned int D;
extern MODE M;
extern unsigned int I;

// External variables defined in `NBody`
extern unsigned int grid_size;
extern float normalising_factor;
extern size_t nbodies_soa_size;
extern size_t activity_map_size;

static nbody_soa nbodies;
static float *force_sum_x;
static float *force_sum_y;
static float *activity_map;

// Function declarations
static void step_cpu() noexcept;
static void allocate_memory() noexcept;

/**
 * Entry point of program for CPU and OPENMP mode
 *
 * @return int The program exit code
 */
int main_cpu() {
    // Allocate any heap memory
    allocate_memory();

    // Initialise N-bodies data
    initialise_data_soa(&nbodies);

    if (I == 0) {
        // Start the visualiser
        initViewer(N, D, M, &step_cpu);
        setNBodyPositions2f(nbodies.x, nbodies.y);
        setActivityMapData(activity_map);
        startVisualisationLoop();
    } else {
        // Perform a fixed number of simulation steps, then output the timing results
        const double start = omp_get_wtime();

        for (unsigned i = 0; i < I; ++i) {
            step_cpu();
        }

        const double seconds = omp_get_wtime() - start;
        const int milliseconds = static_cast<int>((seconds - static_cast<int>(seconds)) * 1000);

        printf("Execution time %d seconds %d milliseconds\n", static_cast<int>(seconds), milliseconds);
    }

    // Free memory
    free(nbodies.x);
    free(nbodies.y);
    free(nbodies.vx);
    free(nbodies.vy);
    free(nbodies.m);
    free(force_sum_x);
    free(force_sum_y);
    free(activity_map);

    return 0;
}

/**
 * Perform the main simulation of the NBody system on the CPU
 */
static void step_cpu() noexcept {
    int i, j;

    // Clear the activity map of previous step
    memset(activity_map, 0, activity_map_size);

    /* Force */
#pragma omp parallel for schedule(static) default(none) shared(N, nbodies, force_sum_x, force_sum_y) if (M == OPENMP)
    for (i = 0; i < static_cast<int>(N); ++i) {
        float sum_x = 0, sum_y = 0;

#pragma omp parallel for schedule(static) default(none) shared(N, nbodies) reduction(+: sum_x, sum_y) if (M == OPENMP)
        for (j = 0; j < static_cast<int>(N); ++j) {
            const float dist_x = nbodies.x[j] - nbodies.x[i];
            const float dist_y = nbodies.y[j] - nbodies.y[i];
            const float mag_add_soft = dist_x * dist_x + dist_y * dist_y + SOFTENING_SQUARE;
            const float m_div_soft = nbodies.m[j] / (mag_add_soft * sqrtf(mag_add_soft));

            sum_x += m_div_soft * dist_x;
            sum_y += m_div_soft * dist_y;
        }

        force_sum_x[i] = sum_x;
        force_sum_y[i] = sum_y;
    }

#pragma omp parallel for schedule(static) default(none) shared(N, D, nbodies, activity_map) if (M == OPENMP)
    for (i = 0; i < static_cast<int>(N); ++i) {
        /* Movement */
        // Calculate position vector, do this first as it depends on current velocity
        nbodies.x[i] += dt * nbodies.vx[i];
        nbodies.y[i] += dt * nbodies.vy[i];

        // Calculate velocity vector, force and acceleration are computed together
        nbodies.vx[i] += dt * G * force_sum_x[i];
        nbodies.vy[i] += dt * G * force_sum_y[i];

        /* compute the position for a body in the `activity_map`
         * and increase the corresponding body count */
        const unsigned int col = static_cast<unsigned int>(nbodies.x[i] * static_cast<float>(D));
        const unsigned int row = static_cast<unsigned int>(nbodies.y[i] * static_cast<float>(D));

        // Do not update `activity_map` if n-body is out of grid area
        if (row < D && col < D) {
            if (M == OPENMP) {
#pragma omp atomic
                ++activity_map[D * row + col];
            } else {
                ++activity_map[D * row + col];
            }
        }
    }

    /* Loop through the `activity_map` to normalise the body counts */
    for (i = 0; i < static_cast<int>(grid_size); ++i) {
        activity_map[i] *= normalising_factor;
    }
}

/**
 * Allocate required memory
 */
static void allocate_memory() noexcept {
    nbodies.x = static_cast<float *>(malloc(nbodies_soa_size));
    nbodies.y = static_cast<float *>(malloc(nbodies_soa_size));
    nbodies.vx = static_cast<float *>(malloc(nbodies_soa_size));
    nbodies.vy = static_cast<float *>(malloc(nbodies_soa_size));
    nbodies.m = static_cast<float *>(malloc(nbodies_soa_size));

    force_sum_x = static_cast<float *>(malloc(nbodies_soa_size));
    force_sum_y = static_cast<float *>(malloc(nbodies_soa_size));

    activity_map = static_cast<float *>(malloc(activity_map_size));
    if (activity_map == nullptr) {
        fprintf(stderr, "error: failed to allocate memory: activity_map");
        exit(EXIT_FAILURE);
    }
}
