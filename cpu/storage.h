#ifndef STORAGE_H
#define STORAGE_H

#include "kernel.h"

typedef struct {
	vect2 pos;
	decimal mass;
} pointMass;

void serialize(const char* filename);
void deserialize(const char* filename);
void createNew(pointMass* points, int pointc, vect2 imgSize, vect2 imgLlcorner);
vect2 totalBeta(vect2 imgPos);
void freeMemory();
#endif
