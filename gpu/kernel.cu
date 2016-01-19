#include "kernel.h"

#define SEED 1234

typedef enum {
	dir_x, dir_y
} direction;

//#define backgroundBeta(imgPoint,dir) (0) //TODO beta from NFW
//#define pointBeta(imgX,imgY,dir) (0) //TODO beta from point mass

__device__ decimal backgroundBetaX(vect2 img) {
	decimal Tc=1;
	decimal e=0.2;
	decimal K=5;
	decimal F = (Tc*Tc+(1.-e)*img.x*img.x+(1.+e)*img.y*img.y);
	return img.x*K*(1.-e)/sqrt(F);
}

__device__ decimal backgroundBetaY(vect2 img) {
	decimal Tc=1;
	decimal e=0.2;
	decimal K=5;
	decimal F = (Tc*Tc+(1.-e)*img.x*img.x+(1.+e)*img.y*img.y);
	return img.y*K*(1.+e)/sqrt(F);
}

__device__ decimal pointBetaX(decimal imgX, decimal imgY) {
	decimal K=1;
	return K*imgX/(imgX*imgX+imgY*imgY);
}

__device__ decimal pointBetaY(decimal imgX, decimal imgY) {
	decimal K=1;
	return K*imgY/(imgX*imgX+imgY*imgY);
}

__constant__ vect2 points[pointN];

__global__ void bang(curandState *state, vect2* source){
	int i = blockDim.x*blockIdx.x+threadIdx.x;
	if(i<N) {
		curandState localState = state[i];
		vect2 img;
		img.x = curand_uniform(&localState)*20.0-10.0;
		img.y = curand_uniform(&localState)*20.0-10.0;
		state[i] = localState;

		decimal x = img.x - backgroundBetaX(img);
		decimal y = img.y - backgroundBetaY(img);
		for(int j=0; j<pointN; j++){
			vect2 pt = points[j];
			x -= pointBetaX(img.x-pt.x, img.y-pt.y); //TODO points into shared? memory?
			y -= pointBetaY(img.x-pt.x, img.y-pt.y);
		}
		source[i].x = x;
		source[i].y = y;
		//source[i] = img;
	}
}


__global__ void setup_kernel(curandState *state){
	int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence 
       number, no offset */
	if(id<N){
    		curand_init(SEED, id, 0, &state[id]);
	}
}
