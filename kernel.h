#ifndef KERNEL_H
#define KERNEL_H

#include <cuda.h>
#include <curand_kernel.h>

#define THREADS_PER_BLOCK 512
#define N 2048 //pocet paprsku
#define pointN 0 //velikost pole points

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

typedef float decimal; //TODO

typedef struct {
	decimal x,y;
} vect2;

__global__ void bang(curandState *state, vect2* source);
__global__ void setup_kernel(curandState *state);

#endif
