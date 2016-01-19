#ifndef KERNEL_H
#define KERNEL_H

#define THREADS_PER_BLOCK 512
#define N 2048 //pocet paprsku
#define pointN 0 //velikost pole points

typedef float decimal; //TODO

typedef struct {
	decimal x,y;
} vect2;

void bang(vect2* source);

#endif
