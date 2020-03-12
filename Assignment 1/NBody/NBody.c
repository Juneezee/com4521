#define _CRT_SECURE_NO_WARNINGS
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include "NBody.h"
#include "NBodyVisualiser.h"
#include "nbody_args.h"
#include "nbody_data.h"

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

    /*for (unsigned int i = 0; i < N; i++) {
        printf("N_Body: %d\n", i);
        printf("x: %f\n", nbodies[i].x);
        printf("y: %f\n", nbodies[i].y);
        printf("vx: %f\n", nbodies[i].vx);
        printf("vy: %f\n", nbodies[i].vy);
        printf("m: %f\n\n", nbodies[i].m);
    }*/

    if (I == 0) {
        // Start the visualiser
        initViewer(N, D, M, &step);
        setNBodyPositions(nbodies);
        setHistogramData(activity_map);
        startVisualisationLoop();
    } else {
        // Perform a fixed number of simulation steps, then output the timing results
        const clock_t start = clock();

        for (unsigned int i = 0; i < I; i++) {
            //
        }

        const clock_t end = clock();

        const float seconds = (float)(end - start) / CLOCKS_PER_SEC;
        const float milliseconds = (seconds - floorf(seconds)) * 1000;

        printf("Execution time: %.0f seconds %.0f milliseconds", seconds, milliseconds);
    }

    // Free memory
    free(nbodies);
    free(activity_map);

    return 0;
}

static void step(void) {
    // TODO: Perform the main simulation of the NBody system

    for (unsigned int i = 0; i < N; i++) {
        /* Force */
        vector sum = { 0, 0 };

        for (unsigned int j = 0; j < N; j++) {
            const vector dist_ij = {
                nbodies[j].x - nbodies[i].x,
                nbodies[j].y - nbodies[i].y
            };
            const float mag_ij = MAGNITUDE(dist_ij.x, dist_ij.y));
            const float soft_norm = powf(mag_ij * mag_ij + SOFTENING * SOFTENING, 1.5f);

            sum.x += nbodies[j].m * dist_ij.x / soft_norm;
            sum.y += nbodies[j].m * dist_ij.y / soft_norm;
        }

        const float GM = G * nbodies[i].m;
        const vector force = { sum.x * GM, sum.y * GM };

        /* Movement */
        // Calculate position vector, do this first as it depends on current velocity
        nbodies[i].x += dt * nbodies[i].vx;
        nbodies[i].y += dt * nbodies[i].vy;

        // Calculate velocity vector, acceleration is also computed here
        const float dt_m = dt / nbodies[i].m;
        nbodies[i].vx += dt_m * force.x;
        nbodies[i].vy += dt_m * force.y;

        /* compute the position for a body in the `activity_map`
         * and increase the corresponding body count */
        const int col = (int)(nbodies[i].x * (float)D);
        const int row = (int)(nbodies[i].y * (float)D);
        const int cell = (int)(D * row + col);
        activity_map[cell] += 1.0f;
    }

    /* Loop through the `activity_map` to normalise the body counts */
    for (unsigned int i = 0, n = D * D; i < n; i++) {
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
