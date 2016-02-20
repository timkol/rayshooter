#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <libconfig.h>

#include "kernel.h"
#include "storage.h"

//settings part
int resX, resY; //number of pixels of source
vect2 sourceSize; //size of source
vect2 sourceLlcorner; //source left low corner coordinates
long long rayCount; //overall number of rays
vect2 imageSize; //size of image
vect2 imageLlcorner; //image left low corner coordinates

int concurrentThreads;
	
long long **hist_source;

void loadConfig(){
	config_t config;
	
	config_init(&config);

	if(!config_read_file(&config, "config")) {
		fprintf(stderr, "%s:%d - %s\n", config_error_file(&config), config_error_line(&config), config_error_text(&config));
		config_destroy(&config);
		//return(EXIT_FAILURE);
	}

	config_lookup_int(&config, "resX", &resX);
	config_lookup_int(&config, "resY", &resY);

	sourceSize.x=16.0;
	sourceSize.y=12.0;
	sourceLlcorner.x=-8.0;
	sourceLlcorner.y=-6.0;

	imageSize.x=20.0;
	imageSize.y=20.0;
	imageLlcorner.x=-10.0;
	imageLlcorner.y=-10.0;

	config_lookup_int64(&config, "rayCount", &rayCount);
//	config_lookup_int(&config, "pointc", &pointc);

	config_lookup_int(&config, "concurrentThreads", &concurrentThreads);

	config_destroy(&config);
}

void histogram(vect2 pos){
	int n,m;
	n=(int) floor((pos.x-sourceLlcorner.x)/sourceSize.x*resX);
	m=(int) floor((pos.y-sourceLlcorner.y)/sourceSize.y*resY);
	if(m<resY && n<resX && m>=0 && n>=0)
		hist_source[m][n]++;
}

void bang(vect2* source){
	vect2 img;
	img.x = (((decimal) rand())/RAND_MAX*imageSize.x)+imageLlcorner.x;
	img.y = (((decimal) rand())/RAND_MAX*imageSize.y)+imageLlcorner.y;

	vect2 beta = totalBeta(img);
	source->x = img.x - beta.x;
	source->y = img.y - beta.y;
}

void simulate(){
	long long cycles;
	cycles=rayCount/concurrentThreads + 1;

	vect2 source_pos;
	for(long long i=0; i<cycles; i++){
		bang(&source_pos);
		histogram(source_pos);
		if(i%10000000==0) {
			fprintf(stderr, "\n\n%i/%i\n\n", i, cycles);
		}
	}
}

void prepareMemory(){
	hist_source=(long long **)malloc(resY*sizeof(long long *));
	for(unsigned int i=0; i<resY; i++){
		hist_source[i]=(long long *)malloc(resX*sizeof(long long));
	}

	for(int i=0; i<resX; i++){
                for(int j=0; j<resY; j++){
			hist_source[j][i]=0;
                }
        }
}

int main(){
	loadConfig();
	prepareMemory();
	fprintf(stderr, "deserializing");
	deserialize("cells.dat");
	fprintf(stderr, "deserializing complete");

	srand(time(NULL));//TODO different seeds on different processes?
	simulate();

	long long val;
	for(int i=0; i<resX; i++){
		for(int j=0; j<resY; j++){
			val = hist_source[j][i];
			if(val<0) val=0;
			printf("%d\t%d\t%d\n", i, j, val);
		}
	}

	for(unsigned int i=0; i<resY; i++){
                free(hist_source[i]);
        }
	free(hist_source);
	freeMemory();
}
