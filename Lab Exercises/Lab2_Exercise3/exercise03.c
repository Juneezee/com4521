#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include "linked_list.h"

#define NUM_STUDENTS 4

struct student {
	char *forename;
	char *surname;
	float average_module_mark;
};

void print_student(const struct student *s);

void main() {
	llitem* ll;
	llitem* start;
	llitem* end;
	unsigned int str_len;

	start = NULL;
	end = NULL;

	/* the ampersand is actually optional */
	print_callback = (void (*)(void*))&print_student;

	FILE *f = NULL;
	f = fopen("students2.bin", "rb"); //read and binary flags
	if (f == NULL) {
		fprintf(stderr, "Error: Could not find students.bin file \n");
		exit(1);
	}

	while (fread(&str_len, sizeof(unsigned int), 1, f) == 1) {
		struct student *s;
		s = (struct student*) malloc(sizeof(struct student));

		s->forename = (char *) malloc(sizeof(char) * str_len);
		fread(s->forename, sizeof(char), str_len, f);

		fread(&str_len, sizeof(unsigned int), 1, f);
		s->surname = (char*)malloc(sizeof(char) * str_len);
		fread(s->surname, sizeof(char), str_len, f);

		fread(&s->average_module_mark, sizeof(float), 1, f);

		if (end == NULL) {
			end = create_linked_list();
			start = end;
		}
		else {
			end = add_to_linked_list(end);
		}

		end->record = (void *) s;
	}

	fclose(f);

	print_items(start);

	ll = start;
	while (ll != NULL) {
		free(((struct student*)ll->record)->forename);
		free(((struct student*)ll->record)->surname);
		free((struct student*)ll->record);
	}

	free_linked_list(start);
}

void print_student(const struct student *s) {
	printf("Student:\n");
	printf("\tForename: %s\n", s->forename);
	printf("\tSurname: %s\n", s->surname);
	printf("\tAverage Module Mark: %.2f\n", s->average_module_mark);
}
