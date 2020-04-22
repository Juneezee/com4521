#include "NBody.h"
#include "nbody_args.h"
#include "nbody_cpu.h"
#include "nbody_gpu.h"

// External variables defined in `nbody_args`
extern unsigned int N;
extern unsigned int D;
extern MODE M;

// Runtime constants
unsigned int grid_size;
float normalising_factor;
size_t nbodies_aos_size;
size_t nbodies_soa_size;
size_t activity_map_size;

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

    // Calculate the constant variables
    grid_size = D * D;
    normalising_factor = static_cast<float>(D) / static_cast<float>(N);
    nbodies_aos_size = sizeof(nbody) * N;
    nbodies_soa_size = sizeof(float) * N;
    activity_map_size = sizeof(float) * grid_size;

    // Run CPU or GPU version, depending on the mode M
    return M == CUDA ? main_gpu() : main_cpu();
}
