#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nbody_data.h"

#define DEFAULT_X ((float)rand() / RAND_MAX)
#define DEFAULT_Y ((float)rand() / RAND_MAX)
#define DEFAULT_VX 0
#define DEFAULT_VY 0
#define DEFAULT_M (1 / (float)N)

#define BUFFER_SIZE 64

// External variable defined in `nbody_args`
extern unsigned int N;
extern char *input_file;

static void generate_random_data(nbody *);
static void read_input_file(nbody *);
static int read_line(FILE *, char[]);
static void validate_commas(char[]);
static void parse_initial_values(nbody *, char[]);
static char *tokenise(char *);
static float str_to_float(const char *);

/**
 * Depending on program arguments, either generate random data
 * or read initial data from file
 *
 * @param nbodies
 */
void initialise_data(nbody *nbodies) {
    if (input_file == NULL) {
        generate_random_data(nbodies);
    } else {
        read_input_file(nbodies);
    }
}

/**
 * Generate random data for N-bodies
 *
 * @param nbodies A pointer to an N-body structure
 */
static void generate_random_data(nbody *nbodies) {
    for (unsigned int i = 0; i < N; ++i) {
        nbodies[i].x = DEFAULT_X;
        nbodies[i].y = DEFAULT_Y;
        nbodies[i].vx = DEFAULT_VX;
        nbodies[i].vy = DEFAULT_VY;
        nbodies[i].m = DEFAULT_M;
    }
}

/**
 * Read initial data from file
 *
 * @param nbodies A pointer to N-bodies structure
 */
static void read_input_file(nbody *nbodies) {
    FILE *f = fopen(input_file, "r");

    if (f == NULL) {
        fprintf(stderr, "error: failed to read file: %s\n", input_file);
        exit(EXIT_FAILURE);
    }

    // One extra byte needed for null character
    char buffer[BUFFER_SIZE + 1];
    unsigned int nbody_count = 0;

    while (read_line(f, buffer)) {
        if (nbody_count < N) {
            validate_commas(buffer);
            parse_initial_values(&nbodies[nbody_count], buffer);
        }

        ++nbody_count;
    }

    fclose(f);

    // Throw error is N supplied != number of bodies in the input file
    if (nbody_count != N) {
        fprintf(stderr,
            "error: argument N (%u) != the number of bodies in the input file (%u)",
            N, nbody_count);
        exit(EXIT_FAILURE);
    }
}

/**
 * Read the given file `f` line by line
 *
 * @param f The file pointer
 * @param buffer The line buffer
 * @return int 0 if EOF, 1 if finished reading a line
 */
static int read_line(FILE *f, char buffer[]) {
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
static void validate_commas(char buffer[]) {
    unsigned int comma_count = 0;

    for (unsigned int i = 0; buffer[i]; ++i) {
        if (buffer[i] == ',') ++comma_count;
    }

    if (comma_count != 4) {
        fprintf(stderr, "error: incorrect number of commas at line: %s\n", buffer);
        exit(EXIT_FAILURE);
    }
}

/**
 * Convert the 5 initial values in the line buffer to float and store in nbody
 *
 * @param nb A pointer to an N-body structure
 * @param buffer The line buffer
 */
static void parse_initial_values(nbody *nb, char buffer[]) {
    const char *x_token = tokenise(buffer);

    // `tokenise` maintains a static pointer to buffer, pass NULL to get the next token
    const char *y_token = tokenise(NULL);
    const char *vx_token = tokenise(NULL);
    const char *vy_token = tokenise(NULL);
    const char *m_token = tokenise(NULL);

    nb->x = x_token == NULL ? DEFAULT_X : str_to_float(x_token);
    nb->y = y_token == NULL ? DEFAULT_Y : str_to_float(y_token);
    nb->vx = vx_token == NULL ? DEFAULT_VX : str_to_float(vx_token);
    nb->vy = vy_token == NULL ? DEFAULT_VY : str_to_float(vy_token);
    nb->m = m_token == NULL ? DEFAULT_M : str_to_float(m_token);
}

/**
 * Finds the next token in a string, using comma as delimiter
 *
 * @param buffer The line buffer
 * @return A pointer to the next token found in buffer
 */
static char *tokenise(char *buffer) {
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
 * Convert tokenised string into float,
 * including validation for overflow and invalid tokens
 *
 * @param token The tokenised string
 * @return res The converted float
 */
static float str_to_float(const char *token) {
    char *end_ptr;
    const float res = strtof(token, &end_ptr);

    /* Case: any trailing characters that are not part of a float.
     *       e.g. accepts 0.6f, but not a single 'f' */
    if (*end_ptr != '\0' && !(strlen(token) > 1 && strlen(end_ptr) == 1 && end_ptr[0] == 'f')) {
        fprintf(stderr, "error: not convertible to float: %s\n", token);
        exit(EXIT_FAILURE);
    }

    if (errno == ERANGE) {
        fprintf(stderr, "error: float overflow: %s\n", token);
        exit(EXIT_FAILURE);
    }

    return res;
}
