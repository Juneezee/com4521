#include <stdlib.h>

#define RAND_SEED 123
#define NUM_VALUES 250

void init_random() {
	srand(RAND_SEED);
}

unsigned short random_ushort() {
	return (unsigned short) rand();
}
