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
decimal pointBetaX(decimal imgX, decimal imgY);
decimal pointBetaY(decimal imgX, decimal imgY);
decimal backgroundBetaX(vect2 img);
decimal backgroundBetaY(vect2 img);

#endif
