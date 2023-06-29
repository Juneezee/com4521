#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

#include "NBody.h"
#include "NBodyVisualiser.h"
#include "nbody_args.h"
#include "nbody_data.h"

// External variables defined in `nbody_args`
extern unsigned int N;
extern unsigned int D;
extern MODE M;
extern unsigned int I;

static nbody_soa nbodies;
static float *force_sum_x;
static float *force_sum_y;
static float *activity_map;

// Function declarations
static void step(void);
static void allocate_memory(void);

/**
 * Entry point of program
 *
 * @param argc The count of the command arguments
 * @param argv An array (of length argc) of the arguments.
 *             The first argument is always the executable name (including path)
 * @return int The program exit code
 */
int main(const int argc, char *argv[]) {
    // Processes the command line arguments
    parse_argv(argc, argv);

    // Allocate any heap memory
    allocate_memory();

    // Initialise N-bodies data
    initialise_data_soa(&nbodies);

    if (I == 0) {
        // Start the visualiser
        initViewer(N, D, M, &step);
        setNBodyPositions2f(nbodies.x, nbodies.y);
        setHistogramData(activity_map);
        startVisualisationLoop();
    } else {
        // Perform a fixed number of simulation steps, then output the timing results
        const double start = omp_get_wtime();

        for (unsigned i = 0; i < I; ++i) {
            step();
        }

        const double end = omp_get_wtime();

        const double seconds = end - start;
        const int milliseconds = (int)((seconds - (int)seconds) * 1000);

        printf("Execution time %d seconds %d milliseconds\n", (int)seconds, milliseconds);
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
 * Perform the main simulation of the NBody system
 */
static void step(void) {
    int i, j;

    // Clear the previous step values
    const unsigned int grid_size = D * D;
    memset(activity_map, 0, grid_size * sizeof(float));

    /* Force */
#pragma omp parallel for schedule(static) default(none) private(i) shared(N, nbodies, force_sum_x, force_sum_y) if (M == OPENMP)
    for (i = 0; i < (int)N; ++i) {
        float sum_x = 0, sum_y = 0;

#pragma omp parallel for schedule(static) default(none) private(j) shared(N, nbodies) reduction(+: sum_x, sum_y) if (M == OPENMP)
        for (j = 0; j < (int)N; ++j) {
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

#pragma omp parallel for schedule(static) default(none) private(i) shared(N, D, nbodies, activity_map) if (M == OPENMP)
    for (i = 0; i < (int)N; ++i) {
        /* Movement */
        // Calculate position vector, do this first as it depends on current velocity
        nbodies.x[i] += dt * nbodies.vx[i];
        nbodies.y[i] += dt * nbodies.vy[i];

        // Calculate velocity vector, force and acceleration are computed together
        nbodies.vx[i] += dt * G * force_sum_x[i];
        nbodies.vy[i] += dt * G * force_sum_y[i];

        /* compute the position for a body in the `activity_map`
         * and increase the corresponding body count */
        const unsigned int col = (unsigned int)(nbodies.x[i] * (float)D);
        const unsigned int row = (unsigned int)(nbodies.y[i] * (float)D);
        
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
    const float normalise = (float)D / (float)N;
    for (i = 0; i < (int)grid_size; ++i) {
        activity_map[i] *= normalise;
    }
}

/**
 * Allocate required memory
 */
static void allocate_memory(void) {
    const size_t size = sizeof(float) * N;
    nbodies.x = (float *)malloc(size);
    nbodies.y = (float *)malloc(size);
    nbodies.vx = (float *)malloc(size);
    nbodies.vy = (float *)malloc(size);
    nbodies.m = (float *)malloc(size);

    force_sum_x = (float *)malloc(size);
    force_sum_y = (float *)malloc(size);

    activity_map = (float *)malloc(sizeof(float) * D * D);
    if (activity_map == NULL) {
        fprintf(stderr, "error: failed to allocate memory: activity_map");
        exit(EXIT_FAILURE);
    }
}
