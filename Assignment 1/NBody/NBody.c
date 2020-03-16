#define _CRT_SECURE_NO_WARNINGS
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

#include "NBody.h"
#include "NBodyVisualiser.h"
#include "nbody_args.h"
#include "nbody_data.h"

// Optimisation: function macros
#define MAGNITUDE(x, y) ((float)sqrtf((x) * (x) + (y) * (y))
#define SOFTENING_FUNC(mag) (powf((mag) * (mag) + SOFTENING * SOFTENING, 1.5f))

// External variables defined in `nbody_args`
extern unsigned int N;
extern unsigned int D;
extern MODE M;
extern unsigned int I;

static nbody *nbodies;
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
    initialise_data(nbodies);

    if (I == 0) {
        // Start the visualiser
        initViewer(N, D, M, &step);
        setNBodyPositions(nbodies);
        setHistogramData(activity_map);
        startVisualisationLoop();
    } else {
        // Perform a fixed number of simulation steps, then output the timing results
        const double start = omp_get_wtime();

        for (unsigned int i = 0; i < I; i++) {
            step();
        }

        const double end = omp_get_wtime();

        const double seconds = end - start;
        const int milliseconds = (int)((seconds - (int)seconds) * 1000);

        printf("Execution time: %d seconds %d milliseconds", (int)seconds, milliseconds);
    }

    // Free memory
    free(nbodies);
    free(activity_map);

    return 0;
}

/**
 * Perform the main simulation of the NBody system
 */
static void step(void) {
    int i;

    for (i = 0; i < (int)N; i++) {
        /* Force */
        vector sum = { 0, 0 };

        for (unsigned int j = 0; j < N; j++) {
            const vector dist_ij = {
                nbodies[j].x - nbodies[i].x,
                nbodies[j].y - nbodies[i].y
            };
            const float mag_ij = MAGNITUDE(dist_ij.x, dist_ij.y));
            const float m_div_soft = nbodies[j].m / SOFTENING_FUNC(mag_ij);

            sum.x += m_div_soft * dist_ij.x;
            sum.y += m_div_soft * dist_ij.y;
        }

        const float GM = G * nbodies[i].m;
        const vector force = { sum.x * GM, sum.y * GM };

        /* Movement */
        // Calculate position vector, do this first as it depends on current velocity
        nbodies[i].x += dt * nbodies[i].vx;
        nbodies[i].y += dt * nbodies[i].vy;

        // Calculate velocity vector, acceleration is also computed here
        const float dt_div_m = dt / nbodies[i].m;
        nbodies[i].vx += dt_div_m * force.x;
        nbodies[i].vy += dt_div_m * force.y;

        /* compute the position for a body in the `activity_map`
         * and increase the corresponding body count */
        const unsigned int col = (unsigned int)(nbodies[i].x * (float)D);
        const unsigned int row = (unsigned int)(nbodies[i].y * (float)D);

        // Do not update `activity_map` if n-body is out of bounds
        if (col <= D && row <= D) {
            activity_map[D * row + col] += 1;
        }
    }

    /* Loop through the `activity_map` to normalise the body counts */
    const int n = (int)(D * D);
#pragma omp parallel for shared(n, activity_map, D, N) if (M == OPENMP)
    for (i = 0; i < n; i++) {
        activity_map[i] *= (float)D / (float)N;
    }
}

/**
 * Allocate required memory
 */
static void allocate_memory(void) {
    nbodies = (nbody *)malloc(sizeof(nbody) * N);
    if (nbodies == NULL) {
        fprintf(stderr, "error: failed to allocate memory: nbodies\n");
        exit(EXIT_FAILURE);
    }

    // malloc does not set memory to zero, we need to read from `activity_map` before writing it
    activity_map = (float *)calloc(D * D, sizeof(float));
    if (activity_map == NULL) {
        fprintf(stderr, "error: failed to allocate memory: activity_map");
        exit(EXIT_FAILURE);
    }
}
