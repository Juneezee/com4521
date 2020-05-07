// Header guards prevent the contents of the header from being defined multiple times where there are circular dependencies
#ifndef __NBODY_HEADER__
#define __NBODY_HEADER__

#define USER_NAME "acb16zje"

#define G                9.8f    // gravitational constant
#define dt               0.01f   // time step
#define SOFTENING        2.0f    // softening parameter to help with numerical instability
#define SOFTENING_SQUARE 4.0f

typedef struct nbody {
    float x, y, vx, vy, m;
} nbody;

typedef struct nbody_soa {
    float *x, *y, *vx, *vy, *m;
} nbody_soa;

typedef enum MODE { CPU, OPENMP, CUDA } MODE;

#endif	//__NBODY_HEADER__
