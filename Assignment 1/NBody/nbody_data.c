#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

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

static void store_initial_values_aos(nbody *, float, float, float, float, float);
static void store_initial_values_soa(nbody_soa *, unsigned int, float, float, float, float, float);
static void read_input_file_aos(nbody *);
static void read_input_file_soa(nbody_soa *);
static int read_line(FILE *, char[]);
static bool validate_commas(const char[]);
static char *tokenise(char *);
static float str_to_float(const char *);

/**
 * Depending on program arguments, either generate random data
 * or read initial data from file
 *
 * @param nbodies A pointer to N-bodies structure (Array of Structures)
 */
void initialise_data_aos(nbody *nbodies) {
    if (input_file == NULL) {
        // Generate random data for N-bodies (Array of Structures)
        for (unsigned int i = 0; i < N; ++i) {
            nbodies[i].x = DEFAULT_X;
            nbodies[i].y = DEFAULT_Y;
            nbodies[i].vx = DEFAULT_VX;
            nbodies[i].vy = DEFAULT_VY;
            nbodies[i].m = DEFAULT_M;
        }
    } else {
        read_input_file_aos(nbodies);
    }
}

/**
 * Depending on program arguments, either generate random data
 * or read initial data from file
 *
 * @param nbodies A pointer to N-bodies structure (Structure of Arrays)
 */
void initialise_data_soa(nbody_soa *nbodies) {
    if (input_file == NULL) {
        // Generate random data for N-bodies (Structure of Arrays)
        for (unsigned int i = 0; i < N; ++i) {
            nbodies->x[i] = DEFAULT_X;
            nbodies->y[i] = DEFAULT_Y;
            nbodies->vx[i] = DEFAULT_VX;
            nbodies->vy[i] = DEFAULT_VY;
            nbodies->m[i] = DEFAULT_M;
        }
    } else {
        read_input_file_soa(nbodies);
    }
}

/**
 * Store the 5 initial values into the given nbody
 *
 * @param body A pointer to an N-body (Array of Structures)
 * @param x The x value
 * @param y The y value
 * @param vx The vx value
 * @param vy The vy value
 * @param m The m value
 */
static void store_initial_values_aos(nbody *body,
    const float x,
    const float y,
    const float vx,
    const float vy,
    const float m) {
    body->x = x;
    body->y = y;
    body->vx = vx;
    body->vy = vy;
    body->m = m;
}

/**
 * Store the 5 initial values into the given nbody
 *
 * @param body A pointer to an N-body structure (Structure of Arrays)
 * @param index The index representing the n-body
 * @param x The x value
 * @param y The y value
 * @param vx The vx value
 * @param vy The vy value
 * @param m The m value
 */
static void store_initial_values_soa(nbody_soa *body,
    const unsigned int index,
    const float x,
    const float y,
    const float vx,
    const float vy,
    const float m) {
    body->x[index] = x;
    body->y[index] = y;
    body->vx[index] = vx;
    body->vy[index] = vy;
    body->m[index] = m;
}

/**
 * Read initial data from file
 *
 * @param nbodies A pointer to N-bodies structure (Array of Structures)
 */
static void read_input_file_aos(nbody *nbodies) {
    FILE *f = fopen(input_file, "r");

    if (f == NULL) {
        fprintf(stderr, "error: failed to read file: %s\n", input_file);
        exit(EXIT_FAILURE);
    }

    // One extra byte needed for null character
    char buffer[BUFFER_SIZE + 1];
    unsigned int nbody_count = 0;

    while (read_line(f, buffer)) {
        if (nbody_count < N && validate_commas(buffer)) {
            // Convert the 5 initial values in the line buffer to float and store in nbody
            // `tokenise` maintains a static pointer to buffer, pass NULL to get the next token
            const char *x_token = tokenise(buffer);
            const char *y_token = tokenise(NULL);
            const char *vx_token = tokenise(NULL);
            const char *vy_token = tokenise(NULL);
            const char *m_token = tokenise(NULL);

            const float x = x_token == NULL ? DEFAULT_X : str_to_float(x_token);
            const float y = y_token == NULL ? DEFAULT_Y : str_to_float(y_token);
            const float vx = vx_token == NULL ? DEFAULT_VX : str_to_float(vx_token);
            const float vy = vy_token == NULL ? DEFAULT_VY : str_to_float(vy_token);
            const float m = m_token == NULL ? DEFAULT_M : str_to_float(m_token);

            store_initial_values_aos((nbody *)nbodies, x, y, vx, vy, m);
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
 * Read initial data from file
 *
 * @param nbodies A pointer to N-bodies structure (Structure of Arrays)
 */
static void read_input_file_soa(nbody_soa *nbodies) {
    FILE *f = fopen(input_file, "r");

    if (f == NULL) {
        fprintf(stderr, "error: failed to read file: %s\n", input_file);
        exit(EXIT_FAILURE);
    }

    // One extra byte needed for null character
    char buffer[BUFFER_SIZE + 1];
    unsigned int nbody_count = 0;

    while (read_line(f, buffer)) {
        if (nbody_count < N && validate_commas(buffer)) {
            // Convert the 5 initial values in the line buffer to float and store in nbody
            // `tokenise` maintains a static pointer to buffer, pass NULL to get the next token
            const char *x_token = tokenise(buffer);
            const char *y_token = tokenise(NULL);
            const char *vx_token = tokenise(NULL);
            const char *vy_token = tokenise(NULL);
            const char *m_token = tokenise(NULL);

            const float x = x_token == NULL ? DEFAULT_X : str_to_float(x_token);
            const float y = y_token == NULL ? DEFAULT_Y : str_to_float(y_token);
            const float vx = vx_token == NULL ? DEFAULT_VX : str_to_float(vx_token);
            const float vy = vy_token == NULL ? DEFAULT_VY : str_to_float(vy_token);
            const float m = m_token == NULL ? DEFAULT_M : str_to_float(m_token);

            store_initial_values_soa((nbody_soa *)nbodies, nbody_count, x, y, vx, vy, m);
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
 * @return bool true if the comma count is exactly 4, otherwise exit with failure
 */
static bool validate_commas(const char buffer[]) {
    unsigned int comma_count = 0;

    for (unsigned int i = 0; buffer[i]; ++i) {
        if (buffer[i] == ',') ++comma_count;
    }

    if (comma_count != 4) {
        fprintf(stderr, "error: incorrect number of commas at line: %s\n", buffer);
        exit(EXIT_FAILURE);
    }

    return true;
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
