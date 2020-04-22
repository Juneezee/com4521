#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "NBody.h"
#include "nbody_args.h"

// User supplied argument variables
unsigned int N;
unsigned int D;
MODE M;
unsigned int I;
char *input_file;

static void print_help() noexcept;
static unsigned int str_to_u_int(const char *) noexcept;

/**
 * Prints help message
 */
static void print_help() noexcept {
    printf("nbody_%s N D M [-i I] [-i input_file]\n", USER_NAME);

    printf("where:\n");
    printf("\tN                Is the number of bodies to simulate.\n");
    printf("\tD                Is the integer dimension of the activity grid. The Grid has D*D locations.\n");
    printf("\tM                Is the operation mode, either 'CPU' or 'OPENMP'\n");
    printf("\t[-i I]           Optionally specifies the number of simulation iterations 'I' to perform. Specifying no value will use visualisation mode.\n");
    printf("\t[-f input_file]  Optionally specifies an input file with an initial N bodies of data. If not specified random data will be created.\n");
}

/**
 * Process the arguments passed by user
 *
 * @param argc The count of the command arguments
 * @param argv An array (of length argc) of the arguments.
 *             The first argument is always the executable name (including path)
 */
void parse_argv(const int argc, char *argv[]) {
    // Need to have at least 3 arguments, N, D, and M
    if (argc < 4) {
        fprintf(stderr, "error: not enough arguments\n");
        print_help();
        exit(EXIT_FAILURE);
    }

    // N and D must be larger than 1
    N = str_to_u_int(argv[1]);
    D = str_to_u_int(argv[2]);
    if (N == 0 || D == 0) {
        fprintf(stderr, "error: N and D must be larger than 0");
        exit(EXIT_FAILURE);
    }
    // Arithmetic overflow will occur when doing (int)(D * D) in `step` function
    if (D >= 46341) {
        fprintf(stderr, "error: overflow will occur if D is too large, try 0 < D < 46341");
        exit(EXIT_FAILURE);
    }

    if (strcmp(argv[3], "CPU") == 0) {
        M = CPU;
    } else if (strcmp(argv[3], "OPENMP") == 0) {
        M = OPENMP;
    } else if (strcmp(argv[3], "CUDA") == 0) {
        M = CUDA;
    } else {
        fprintf(stderr, "error: invalid operation mode: %s\n", argv[3]);
        exit(EXIT_FAILURE);
    }

    // Optional argv, user should be able to specify flags in any order
    // If a flag is specified more than 1 time, only the last one will be used
    for (int i = 4; i < argc; i += 2) {
        switch (argv[i][1]) {
            case 'i':
                I = str_to_u_int(argv[i + 1]);
                break;
            case 'f':
                input_file = argv[i + 1];

                // -f specified but no input file provided
                if (input_file == nullptr) {
                    fprintf(stderr, "error: an input file is required for option: -f\n");
                    exit(EXIT_FAILURE);
                }
                break;
            default:
                fprintf(stderr, "error: unknown option `%s`\n", argv[i]);
                exit(EXIT_FAILURE);
        }
    }
}

/**
 * Convert a string into unsigned int,
 * including validation for overflow and invalid inputs
 *
 * @param str The string to convert into unsigned int
 * @return res The converted unsigned int
 */
static unsigned int str_to_u_int(const char *str) noexcept {
    // Case: `str` is null, empty or a space
    if (str == nullptr || str[0] == '\0' || isspace(str[0])) {
        fprintf(stderr, "error: integer is empty\n");
        exit(EXIT_FAILURE);
    }

    char *end_ptr;

    const unsigned long res = strtoul(str, &end_ptr, 10);

    // Case: any trailing characters that are not part of the number
    if (*end_ptr != '\0') {
        fprintf(stderr, "error: not convertible to integer: %s\n", str);
        exit(EXIT_FAILURE);
    }

    // res > INT_MAX because OpenMP 2.0 requires `int` for loop counter,
    // so N, D, I need to be explicitly cast to `int`, e.g. i < (int)N
    if (res > INT_MAX || errno == ERANGE) {
        fprintf(stderr, "error: integer overflow: %s\n", str);
        exit(EXIT_FAILURE);
    }

    return static_cast<unsigned int>(res);
}
