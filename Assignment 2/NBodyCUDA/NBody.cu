#include "NBody.h"
#include "nbody_args.h"
#include "nbody_cpu.h"
#include "nbody_gpu.h"

extern MODE M;

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

    // Run CPU or GPU version, depending on the mode M
    return M == CUDA ? main_gpu() : main_cpu();
}
