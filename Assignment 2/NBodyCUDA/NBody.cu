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

static nbody *nbodies;
static float *activity_map;

// Function declarations
static void step() noexcept;
static void allocate_memory() noexcept;

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

        for (unsigned i = 0; i < I; ++i) {
            step();
        }

        const double end = omp_get_wtime();

        const double seconds = end - start;
        const int milliseconds = static_cast<int>((seconds - static_cast<int>(seconds)) * 1000);

        printf("Execution time %d seconds %d milliseconds\n", static_cast<int>(seconds), milliseconds);
    }

    // Free memory
    free(nbodies);
    free(activity_map);

    return 0;
}

/**
 * Perform the main simulation of the NBody system
 */
static void step() noexcept {
    int i;

    // Clear the previous step values
    const unsigned int grid_size = D * D;
    memset(activity_map, 0, grid_size * sizeof(float));

#pragma omp parallel for schedule(static) default(none) shared(N, D, nbodies, activity_map) if (M == OPENMP)
    for (i = 0; i < static_cast<int>(N); ++i) {
        /* Force */
        float sum_x = 0, sum_y = 0;

        for (unsigned int j = 0; j < N; ++j) {
            const float dist_x = nbodies[j].x - nbodies[i].x;
            const float dist_y = nbodies[j].y - nbodies[i].y;
            const float mag_add_soft = dist_x * dist_x + dist_y * dist_y + SOFTENING_SQUARE;
            const float m_div_soft = nbodies[j].m / (mag_add_soft * sqrtf(mag_add_soft));

            sum_x += m_div_soft * dist_x;
            sum_y += m_div_soft * dist_y;
        }

        /* Movement */
        // Calculate position vector, do this first as it depends on current velocity
        nbodies[i].x += dt * nbodies[i].vx;
        nbodies[i].y += dt * nbodies[i].vy;

        // Calculate velocity vector, force and acceleration are computed together
        nbodies[i].vx += dt * G * sum_x;
        nbodies[i].vy += dt * G * sum_y;

        /* compute the position for a body in the `activity_map`
         * and increase the corresponding body count */
        const unsigned int col = static_cast<unsigned int>(nbodies[i].x * static_cast<float>(D));
        const unsigned int row = static_cast<unsigned int>(nbodies[i].y * static_cast<float>(D));
        const unsigned int cell = static_cast<unsigned int>(D * row + col);

        // Do not update `activity_map` if n-body is out of grid area
        if (cell < grid_size) {
            if (M == OPENMP) {
#pragma omp atomic
                ++activity_map[cell];
            } else {
                ++activity_map[cell];
            }
        }
    }

    /* Loop through the `activity_map` to normalise the body counts */
    const float normalise = static_cast<float>(D) / static_cast<float>(N);
    for (i = 0; i < static_cast<int>(grid_size); ++i) {
        activity_map[i] *= normalise;
    }
}

/**
 * Allocate required memory
 */
static void allocate_memory() noexcept {
    nbodies = static_cast<nbody *>(malloc(sizeof(nbody) * N));
    if (nbodies == nullptr) {
        fprintf(stderr, "error: failed to allocate memory: nbodies\n");
        exit(EXIT_FAILURE);
    }

    activity_map = static_cast<float *>(malloc(sizeof(float) * D * D));
    if (activity_map == nullptr) {
        fprintf(stderr, "error: failed to allocate memory: activity_map");
        exit(EXIT_FAILURE);
    }
}
