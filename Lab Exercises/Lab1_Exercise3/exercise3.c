#include <stdio.h>
#include "random.h"

float value[NUM_VALUES];

int main() {
    float sum = 0;
    float average;
    float min = 0;
    float max = 0;

    unsigned char i = 0;

    init_random();

    for (i = 0; i < NUM_VALUES; i++) {
        value[i] = random_float();
        sum += value[i];

        printf("%d: %.0f\n", i, value[i]);
    }

    average = sum / NUM_VALUES;

    for (i = 0; i < NUM_VALUES; i++) {
        value[i] -= average;
        min = value[i] < min ? value[i] : min;
        max = value[i] > max ? value[i] : max;
    }

    printf("Sum = %.0f\n", sum);
    printf("Average = %.0f\n", average);
    printf("Min = %.0f\n", min);
    printf("Max = %.0f\n", max);

    return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started:
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
