#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <libconfig.h>
#include <cuda_runtime_api.h>

#include "kernel.h"

//settings part
int resX, resY; //number of pixels of source
vect2 sourceSize; //size of source
vect2 sourceLlcorner; //source left low corner coordinates
long long rayCount; //overall number of rays
int pointc; //number of mass points
vect2 imageSize; //size of image

int threadsPerBlock;
int concurrentThreads;
	
vect2 *h_source, *h_image;
vect2 *d_source, *d_image;

curandState *devStates;

long long **hist_source;

void estimateDeviceOccupancy(){
//	int blockSize, gridSize;
//	cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, (void*)bang, 0, 0);
//	printf("%d, %d\n", gridSize, blockSize);
}

void loadConfig(){
	config_t config;
	
	config_init(&config);

	if(!config_read_file(&config, "config")) {
		fprintf(stderr, "%s:%d - %s\n", config_error_file(&config), config_error_line(&config), config_error_text(&config));
		config_destroy(&config);
		//return(EXIT_FAILURE);
	}

//	resX=1000;
//	resY=1000;
//	rayCount=10000;
//	pointc=0;
	config_lookup_int(&config, "resX", &resX);
	config_lookup_int(&config, "resY", &resY);

	sourceSize.x=16.0;
	sourceSize.y=12.0;
	sourceLlcorner.x=-8.0;
	sourceLlcorner.y=-6.0;

	config_lookup_int64(&config, "rayCount", &rayCount);
	config_lookup_int(&config, "pointc", &pointc);

	imageSize.x=20.0;
	imageSize.y=20.0;

	int forceCudaSettings = 0;
	config_lookup_bool(&config, "forceCudaSettings", &forceCudaSettings);

	if(forceCudaSettings) {
		config_lookup_int(&config, "threadsPerBlock", &threadsPerBlock);
		config_lookup_int(&config, "concurrentThreads", &concurrentThreads);
	}
	else {
		threadsPerBlock=128;
		concurrentThreads=2048;
	}

	config_destroy(&config);
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
	int m, n;
	for(int i=0; i<concurrentThreads; i++){
//		printf("%i\t%f\t%f\n", i, h_source[i].x, h_source[i].y);
		n=(int) floor((h_source[i].x-sourceLlcorner.x)/sourceSize.x*resX);
		m=(int) floor((h_source[i].y-sourceLlcorner.y)/sourceSize.y*resY);
//		m=((unsigned int)floor(h_source[i].y))%resY;
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
		if(i%1000==0) {
			fprintf(stderr, "\n\n%i/%i\n\n", i, cycles);
		}
	}
}

int main(){
	loadConfig();

	hist_source=(long long **)malloc(resY*sizeof(long long *));
	for(unsigned int i=0; i<resY; i++){
		hist_source[i]=(long long *)malloc(resX*sizeof(long long));
	}
	h_source = (vect2*)malloc(concurrentThreads*sizeof(vect2));
//	h_image = (vect2*)malloc(rayCount*sizeof(vect2));
	CUDA_CALL(cudaMalloc((void **)&d_source, concurrentThreads*sizeof(vect2)));
//	cudaMalloc((void **)&d_image, rayCount*sizeof(vect2));
	CUDA_CALL(cudaMalloc((void **)&devStates, concurrentThreads*sizeof(curandState)));

	CUDA_CALL(cudaMemset(d_source, 0, concurrentThreads*sizeof(vect2)));
	memset(h_source, 0, concurrentThreads*sizeof(vect2));
//	memset(hist_source, 0, resX*resY*sizeof(hist_source[0][0]));
	for(int i=0; i<resX; i++){
                for(int j=0; j<resY; j++){
			hist_source[j][i]=0;
                }
//              printf("\n");
        }
//	generateRandomImage(h_image);

	setup_kernel<<<(concurrentThreads+threadsPerBlock-1)/threadsPerBlock, threadsPerBlock>>>(devStates);
	cudaThreadSynchronize();
//	estimateDeviceOccupancy();
	simulate();
//	cudaMemcpy(d_image, h_image, rayCount*sizeof(vect2), cudaMemcpyHostToDevice);


//	printf("%f\t%f\n", h_image[0].x, h_image[0].y);

	long long val;
	for(int i=0; i<resX; i++){
		for(int j=0; j<resY; j++){
			val = hist_source[j][i];
			if(val<0) val=0;
			printf("%d\t%d\t%d\n", i, j, val);
		}
//		printf("\n");
	}

	CUDA_CALL(cudaFree(d_source));
//	cudaFree(d_image);
//	free(h_image);
	free(h_source);
}
