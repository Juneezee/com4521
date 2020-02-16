#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

#define BUFFER_SIZE 32

int readLine(char buffer[]);

int main() {
    float in_value, sum;
	char buffer [BUFFER_SIZE];
	char command [4];
    sum = 0;

	printf("Welcome to basic COM4521 calculator\n");

	FILE* f = fopen("commands.calc", "r");

    while (readLine(f, buffer)) {
		//4.5 Check that the line contains 3 letters and a spaceextract
		if (!(isalpha(buffer[0]) && isalpha(buffer[1]) && isalpha(buffer[2]) && buffer[3] == ' ')) {
			fprintf(stderr, "Incorrect command format\n");
			continue;
		}

		//4.6 Extract the command and in_value using sscanf
		sscanf(buffer, "%s %f", command, &in_value);

		if (strncmp(command, "add", 2) == 0) { //4.7 Change condition to check command to see if it is "add"
			sum += in_value;
		//4.8 Add else if conditions for sub, mul and div
		} else if (strncmp(command, "sub", 2) == 0) {
			sum -= in_value;
		} else if (strncmp(command, "mul", 2) == 0) {
			sum *= in_value;
		} else if (strncmp(command, "div", 2) == 0) {
			sum /= in_value;
		} else {
			fprintf(stderr, "Unknown command\n");
			continue;
		}

		if (isnan(sum)) {
			sum = 0;
		}
	}

	printf("\tFinal Sum is %f\n", sum);

    return 0;
}

int readLine(FILE *f, char buffer[])  {
	int i = 0;
	char c = 0;
	while ((c = getc(f)) != '\n') {
		if (c == EOF) {
			return 0;
		}

        //4.1 Add character to buffer
		buffer[i++] = c;

		//4.2 Check index for overflow
		if (i == BUFFER_SIZE) {
			fprintf(stderr, "Buffer size limit reached");
			exit(1);
		}
	}
	//4.3 Ensure the buffer is correctly terminated
	buffer[i] = '\0';

	//4.4 Return 0 if buffer = "exit" otherwise return 1
	if (strncmp(buffer, "exit", 4) == 0) {
		return 0;
	} else {
		return 1;
	}
}
