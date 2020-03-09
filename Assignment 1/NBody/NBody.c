#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "NBody.h"
#include "NBodyVisualiser.h"

#define BUFFER_SIZE 64

#define USER_NAME "acb16zje"

// Functions declarations
void step(void);
void print_help(void);
void parse_argv(int, char *[]);
unsigned int str_to_u_int(char *);
void generate_random_data(nbody *);
void read_input_file(nbody *);
int read_line(FILE *, char[]);
void validate_commas(char[]);
char *tokenise(char *);
void parse_initial_values(nbody *, char[]);

// Arguments variable declaration
static unsigned int N;
unsigned int D;
MODE M;
unsigned int I;
char *input_file;

/**
 * Entry point of program
 *
 * @param argc The count of the command arguments
 * @param argv An array (of length argc) of the arguments.
 *             The first argument is always the executable name (including path)
 * @return int The program exit code
 */
int main(const int argc, char *argv[]) {
    // Need to have at least 3 arguments, N, D, and M
    if (argc < 4) {
        fprintf(stderr, "error: not enough arguments\n");
        print_help();
        return 1;
    }

    // Processes the command line arguments
    parse_argv(argc, argv);

    // TODO: Allocate any heap memory
    nbody *nbodies = (nbody *)malloc(sizeof(nbody) * N);
    if (nbodies == NULL) {
        fprintf(stderr, "error: failed to allocate memory: nbodies\n");
        return 1;
    }

    // Depending on program arguments, either generate random data or read initial data from file
    if (input_file == NULL) {
        for (unsigned int i = 0; i < N; i++) {
            generate_random_data(&nbodies[i]);
        }
    } else {
        read_input_file(nbodies);
    }

    // TODO:
    if (I == 0) {
        // Start the visualiser
    } else {
        // Perform a fixed number of simulation steps (then output the timing results).
    }

    // Free memory
    free(nbodies);

    return 0;
}

void step(void) {
    // TODO: Perform the main simulation of the NBody system
}

/**
 * Prints help message
 */
void print_help(void) {
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
    N = str_to_u_int(argv[1]);
    D = str_to_u_int(argv[2]);

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
unsigned int str_to_u_int(char *str) {
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
        fprintf(stderr, "error: integer not in range: %s\n", str);
        exit(EXIT_FAILURE);
    }

    return (unsigned int)res;
}

/**
 * Generate random data for an N-body
 *
 * @param nb A pointer to an N-body structure
 */
void generate_random_data(nbody *nb) {
    nb->x = (float)rand() / RAND_MAX;
    nb->y = (float)rand() / RAND_MAX;
    nb->vx = 0;
    nb->vy = 0;
    nb->m = 1 / (float)N;
}

/**
 * Read initial data from file
 *
 * @param nbodies A pointer to N-bodies structure
 */
void read_input_file(nbody *nbodies) {
    FILE *f = fopen(input_file, "r");

    if (f == NULL) {
        fprintf(stderr, "error: failed to read file: %s\n", input_file);
        exit(EXIT_FAILURE);
    }

    // One extra byte needed for null character
    char buffer[BUFFER_SIZE + 1];
    unsigned int nbody_count = 0;

    while (read_line(f, buffer)) {
        validate_commas(buffer);

        parse_initial_values(&nbodies[nbody_count], buffer);
        nbody_count++;
    }

    // Throw error is N supplied != number of bodies in the input file
    if (N != nbody_count) {
        fprintf(stderr,
            "error: argument N (%u) != the number of bodies in the input file (%u)",
            N, nbody_count);
        exit(EXIT_FAILURE);
    }

    fclose(f);
}

/**
 * Read the given file `f` line by line
 *
 * @param f The file pointer
 * @param buffer The line buffer
 * @return int 0 if EOF, 1 if finished reading a line
 */
int read_line(FILE *f, char buffer[]) {
    int i = 0;
    char c;

    while ((c = (char)getc(f)) != EOF) {
        // Case 1: ignore any line starting with '#'
        // Case 2: ignore any blank line
        if (i == 0 && (c == '#' || c == '\n')) {
            while (c != '\n') {
                c = (char)getc(f);
            }
        } else {
            // End of a useful line, ensure the buffer is correctly terminated
            if (c == '\n') {
                buffer[i] = '\0';
                return 1;
            }

            // Skip spaces so `tokenise` will return NULL for missing values
            if (c == ' ') continue;

            buffer[i++] = c;
            if (i == BUFFER_SIZE) {
                fprintf(stderr, "error: Maximum line length of %d reached\n", BUFFER_SIZE);
                exit(EXIT_FAILURE);
            }
        }
    }

    return 0;
}

/**
 * Validate the line containing the initial values. It should only contain 4 commas.
 *
 * @param buffer The line buffer
 */
void validate_commas(char buffer[]) {
    unsigned int comma_count = 0;

    for (unsigned int i = 0; i < strlen(buffer); i++) {
        if (buffer[i] == ',') comma_count++;
    }

    if (comma_count != 4) {
        fprintf(stderr, "error: incorrect number of commas at line: %s\n", buffer);
        exit(EXIT_FAILURE);
    }
}

/**
 * Finds the next token in a string, using comma as delimiter
 *
 * @param buffer The line buffer
 * @return A pointer to the next token found in buffer
 */
char *tokenise(char *buffer) {
    static char *buffer_start = NULL;

    if (buffer != NULL) buffer_start = buffer;

    // see if we have reached the end of the line
    if (buffer_start == NULL || *buffer_start == '\0') return NULL;

    // return the number of characters that are not delimiters
    const unsigned int n = strcspn(buffer_start, ",");

    // return token as NULL for consecutive delimiters
    if (n == 0) {
        buffer_start += 1;
        return NULL;
    }

    // save start of this token
    char *p = buffer_start;

    // bump past the delimiters
    buffer_start += n;

    // remove the delimiters
    if (*buffer_start != '\0') *buffer_start++ = '\0';

    return p;
}

/**
 * Convert the 5 initial values in the line buffer to float and store in nbody
 *
 * @param nb A pointer to an N-body structure
 * @param buffer The line buffer
 */
void parse_initial_values(nbody *nb, char buffer[]) {
    const char *x_token = tokenise(buffer);

    // `tokenise` maintains a static pointer to buffer, pass NULL to get the next token
    const char *y_token = tokenise(NULL);
    const char *vx_token = tokenise(NULL);
    const char *vy_token = tokenise(NULL);
    const char *m_token = tokenise(NULL);

    nb->x = x_token == NULL ? (float)rand() / RAND_MAX : strtof(x_token, NULL);
    nb->y = y_token == NULL ? (float)rand() / RAND_MAX : strtof(y_token, NULL);
    nb->vx = vx_token == NULL ? 0 : strtof(vx_token, NULL);
    nb->vy = vy_token == NULL ? 0 : strtof(vy_token, NULL);
    nb->m = m_token == NULL ? 1 / (float)N : strtof(m_token, NULL);
}
