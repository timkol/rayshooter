#ifndef KERNEL_H
#define KERNEL_H

typedef float decimal; //TODO

typedef struct {
	decimal x,y;
} vect2;

vect2 pointBeta(decimal imgX, decimal imgY);
vect2 backgroundBeta(vect2 img);

#endif
