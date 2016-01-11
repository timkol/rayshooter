#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "kernel.h"

//settings part
unsigned int resX, resY; //number of pixels of source
unsigned long long rayCount; //overall number of rays
unsigned int pointc; //number of mass points
vect2 imageSize; //size of image

unsigned int threadsPerBlock;
unsigned int concurrentThreads;
	
vect2 *h_source, *h_image;
vect2 *d_source, *d_image;

curandState *devStates;

unsigned long long **hist_source;

void loadConfig(){
	resX=1000;
	resY=1000;
	rayCount=10000;
	pointc=0;

	imageSize.x=20.0;
	imageSize.y=20.0;

	threadsPerBlock=128;
	concurrentThreads=2048;
}

//TODO
void generateRandomImage(vect2* image){
	srand(time(NULL));	
	for(int i=0; i<rayCount; i++){
		image[i].x=(((decimal) rand())/RAND_MAX*20.0)-10;
		image[i].y=(((decimal) rand())/RAND_MAX*20.0)-10;
	}
}

void histogram(){//TODO
	unsigned int m, n;
	for(unsigned int i=0; i<concurrentThreads; i++){
		printf("%i\t%f\t%f\n", i, h_source[i].x, h_source[i].y);
		n=((unsigned int)floor(h_source[i].x))%resX;
		m=((unsigned int)floor(h_source[i].y))%resY;
		hist_source[m][n]++;
	}
}

void simulate(){
	unsigned long long cycles;
	cycles=rayCount/concurrentThreads + 1;

	for(unsigned long long i=0; i<cycles; i++){
		bang<<<(concurrentThreads+threadsPerBlock-1)/threadsPerBlock, threadsPerBlock>>>(devStates, d_source);
		cudaThreadSynchronize();
	
		cudaMemcpy(h_source, d_source, concurrentThreads*sizeof(vect2), cudaMemcpyDeviceToHost);
		histogram();
		printf("\n\n%i\n\n", i);
	}
}

int main(){
	loadConfig();

	hist_source=(unsigned long long **)malloc(resY*sizeof(unsigned long long *));
	for(unsigned int i=0; i<resY; i++){
		hist_source[i]=(unsigned long long *)malloc(resX*sizeof(unsigned long long));
	}
	h_source = (vect2*)malloc(concurrentThreads*sizeof(vect2));
//	h_image = (vect2*)malloc(rayCount*sizeof(vect2));
	CUDA_CALL(cudaMalloc((void **)&d_source, concurrentThreads*sizeof(vect2)));
//	cudaMalloc((void **)&d_image, rayCount*sizeof(vect2));
	CUDA_CALL(cudaMalloc((void **)&devStates, concurrentThreads*sizeof(curandState)));

	CUDA_CALL(cudaMemset(d_source, 0, concurrentThreads*sizeof(vect2)));
	memset(h_source, 0, concurrentThreads*sizeof(vect2));
//	generateRandomImage(h_image);

	setup_kernel<<<(concurrentThreads+threadsPerBlock-1)/threadsPerBlock, threadsPerBlock>>>(devStates);
	cudaThreadSynchronize();
	simulate();
//	cudaMemcpy(d_image, h_image, rayCount*sizeof(vect2), cudaMemcpyHostToDevice);


//	printf("%f\t%f\n", h_image[0].x, h_image[0].y);

	for(unsigned int i=0; i<resX; i++){
		for(unsigned int j=0; j<resY; j++){
//			printf("%i\t", hist_source[j][i]);
		}
//		printf("\n");
	}

	CUDA_CALL(cudaFree(d_source));
//	cudaFree(d_image);
//	free(h_image);
	free(h_source);
}
