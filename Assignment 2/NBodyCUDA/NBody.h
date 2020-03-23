//Header guards prevent the contents of the header from being defined multiple times where there are circular dependencies
#ifndef __NBODY_HEADER__
#define __NBODY_HEADER__

#define G			     1.0f    // gravitational constant (not the actual value of G, a value of G used to avoid issues with numeric precision)
#define dt			     0.01f   // time step
#define SOFTENING	     2.0f    // softening parameter to help with numerical instability
#define SOFTENING_SQUARE 4.0f

typedef struct nbody {
    float x, y, vx, vy, m;
} nbody;

typedef struct nbody_soa {
    float *x, *y, *vx, *vy, *m;
} nbody_soa;

typedef enum MODE { CPU, OPENMP, CUDA } MODE;

#endif	//__NBODY_HEADER__
