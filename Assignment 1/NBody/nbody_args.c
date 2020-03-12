#define _CRT_SECURE_NO_WARNINGS
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

static void print_help(void);
static unsigned int str_to_u_int(char *);

/**
 * Prints help message
 */
static void print_help(void) {
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
void parse_argv(int argc, char *argv[]) {
    // Need to have at least 3 arguments, N, D, and M
    if (argc < 4) {
        fprintf(stderr, "error: not enough arguments\n");
        print_help();
        exit(EXIT_FAILURE);
    };;

    // N and D must be larger than 1
    N = str_to_u_int(argv[1]);
    D = str_to_u_int(argv[2]);
    if (N == 0 || D == 0) {
        fprintf(stderr, "error: N and D must be larger than 0");
        exit(EXIT_FAILURE);
    }

    if (strncmp(argv[3], "CPU", sizeof(char *)) == 0) {
        M = CPU;
    } else if (strncmp(argv[3], "OPENMP", sizeof(char *)) == 0) {
        M = OPENMP;
    } else {
        fprintf(stderr, "error: invalid operation mode: %s\n", argv[3]);
        exit(EXIT_FAILURE);
    }

    // Optional argv, user should be able to specify flags in any order
    // If a flag is specified more than 1 time, only the last one will be used
    for (int i = 4; i < argc; i += 2) {
        const char *flag = argv[i];
        const unsigned int size = sizeof flag;

        if (strncmp(flag, "-i", size) == 0) {
            I = str_to_u_int(argv[i + 1]);
        } else if (strncmp(flag, "-f", size) == 0) {
            input_file = argv[i + 1];

            // -f specified but no input file provided
            if (input_file == NULL) {
                fprintf(stderr, "error: an input file is required for option: -f\n");
                exit(EXIT_FAILURE);
            }
        } else {
            fprintf(stderr, "error: unknown option `%s`\n", flag);
            exit(EXIT_FAILURE);
        }
    }
}

/**
 * Convert a string into unsigned int,
 * including validation for overflow and invalid inputs
 *
 * @param str The string to convert into unsigned int
 * @return unsigned int The converted unsigned int
 */
static unsigned int str_to_u_int(char *str) {
    // Case: `str` is null, empty or a space
    if (str == NULL || str[0] == '\0' || isspace(str[0])) {
        fprintf(stderr, "error: integer is empty\n");
        exit(EXIT_FAILURE);
    }

    char *end_ptr;

    // Need to read as signed to check for negative inputs
    const long long res = strtoll(str, &end_ptr, 10);

    // Case: any trailing characters that are not part of the number
    if (*end_ptr != '\0') {
        fprintf(stderr, "error: integer not convertible: %s\n", str);
        exit(EXIT_FAILURE);
    }

    // Case: overflow of unsigned int
    if (res < 0 || res > UINT_MAX || errno == ERANGE) {
        fprintf(stderr, "error: integer overflow: %s\n", str);
        exit(EXIT_FAILURE);
    }

    return (unsigned int)res;
}