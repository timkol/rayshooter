#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "kernel.h"
#include "storage.h"

typedef struct {
	vect2 extContrib; //background included
	int pointc;
	pointMass *points;
} cell;

cell **pointMatrix;
int matrixDimX, matrixDimY; //dimensions of pointMatrix array
int tolX, tolY; //how many cells on each side are near neighbourhood (0 means just the cell itself)
vect2 imageSize, imageLlcorner;

void allocateMemory();

void serialize(const char* filename) {
	FILE *f;
	f = fopen(filename, "w");
	
	fprintf(f, "%x %x %x %x %a %a %a %a\n", matrixDimX, matrixDimY, tolX, tolY, imageSize.x, imageSize.y, imageLlcorner.x, imageLlcorner.y);

	cell pt;
	pointMass pm;
	for(int i=0; i<matrixDimY; i++){
		for(int j=0; j<matrixDimX; j++){
			pt = pointMatrix[i][j];
			fprintf(f, "%a %a %x\n", pt.extContrib.x, pt.extContrib.y, pt.pointc);
			for(int k=0; k<pt.pointc; k++){
				pm = pt.points[k];
				fprintf(f, "%a %a %a\n", pm.pos.x, pm.pos.y, pm.mass);
			}
		}
	}

	fclose(f);
}

void deserialize(const char* filename) {
	FILE *f;
	f = fopen(filename, "r");
	
	fscanf(f, "%x %x %x %x %a %a %a %a", &matrixDimX, &matrixDimY, &tolX, &tolY, &imageSize.x, &imageSize.y, &imageLlcorner.x, &imageLlcorner.y);

	allocateMemory();

	cell pt;
	pointMass pm;
	for(int i=0; i<matrixDimY; i++){
		for(int j=0; j<matrixDimX; j++){
			fscanf(f, "%a %a %x", &pt.extContrib.x, &pt.extContrib.y, &pt.pointc);
			pt.points = (pointMass *)malloc(pt.pointc*sizeof(pointMass));
			for(int k=0; k<pt.pointc; k++){
				fscanf(f, "%a %a %a", &pm.pos.x, &pm.pos.y, &pm.mass);
				pt.points[k] = pm;
			}
			pointMatrix[i][j] = pt;
		}
	}

	fclose(f);
	fprintf(stderr, "%f %f\n", imageSize.x, imageSize.y);
}

vect2 totalBeta(vect2 imgPos) {//TODO interpolation and near points
	int n=(int) floor((imgPos.x-imageLlcorner.x)/imageSize.x*matrixDimX);
	int m=(int) floor((imgPos.y-imageLlcorner.y)/imageSize.y*matrixDimY);
//	if(m<matrixDimY && n<matrixDimX && m>=0 && n>=0){
//		fprintf(stderr, "%d %d %f %f\n", m, n, imgPos.x, imgPos.y);
//		exit;
//	}
		return pointMatrix[m][n].extContrib;
/*	}
	vect2 tmp;
	tmp.x=0;
	tmp.y=0;
	return tmp; //TODO just debugging
*/
}

void allocateMemory(){
	//allocate memory
        pointMatrix = (cell **)malloc(matrixDimY*sizeof(cell *));
        for(int i=0; i<matrixDimY; i++){
                pointMatrix[i] = (cell *)malloc(matrixDimX*sizeof(cell));
        }

        //make empty
        for(int m=0; m<matrixDimY; m++){
                for(int n=0; n<matrixDimX; n++){
                        pointMatrix[m][n].extContrib.x = 0;
                        pointMatrix[m][n].extContrib.y = 0;
                        pointMatrix[m][n].pointc = 0;
                }
        }

}

void freeMemory(){
	for(int i=0; i<matrixDimY; i++){
		for(int j=0; j<matrixDimX; j++){
			free(pointMatrix[i][j].points);
		}
		free(pointMatrix[i]);
	}
	free(pointMatrix);
}

void createNew(pointMass* points, int pointc, vect2 imgSize, vect2 imgLlcorner) {
	imageSize = imgSize;
	imageLlcorner = imgLlcorner;

	matrixDimX=1000; //TODO user defined values
	matrixDimY=1000;
	tolX=10;
	tolY=10;
/*	//calculate cell dimensions
	vect2 minDist;
	minDist.x=0;
	minDist.y=0;
	decimal temp;
	for(int i=0; i<pointc; i++){
		for(int j=0; j<pointc; j++){
			if(i==j) continue;
			temp = fabs(points[i].pos.x-points[j].pos.x);
			if(temp < minDist.x) minDist.x = temp;
			temp = fabs(points[i].pos.y-points[j].pos.y);
			if(temp < minDist.y) minDist.y = temp;
		}
	}
	int calcResX = (int) ceil(imageSize.x/minDist.x);
	int calcResY = (int) ceil(imageSize.y/minDist.y);
	if(matrixDimX < calcResX) matrixDimX = calcResX;
	if(matrixDimY < calcResY) matrixDimY = calcResY;
*/	
	allocateMemory();

	//populate
	int n,m;
	vect2 pos;
	for(int i=0; i<pointc; i++){
		pos = points[i].pos;
        	n=(int) floor((pos.x-imageLlcorner.x)/imageSize.x*matrixDimX);
        	m=(int) floor((pos.y-imageLlcorner.y)/imageSize.y*matrixDimY);
        	if(m<matrixDimY && n<matrixDimX && m>=0 && n>=0){
			pointMatrix[m][n].pointc++;
		}
	}
	int **tmpCounter;
	tmpCounter = (int **)malloc(matrixDimY*sizeof(int *));
	for(int i=0; i<matrixDimY; i++){
		tmpCounter[i] = (int *)malloc(matrixDimX*sizeof(int));
		for(int j=0; j<matrixDimX; j++){
			tmpCounter[i][j] = 0;
			pointMatrix[i][j].points = (pointMass *)malloc(pointMatrix[i][j].pointc*sizeof(pointMass));
		}
	}
	for(int i=0; i<pointc; i++){
		pos = points[i].pos;
        	n=(int) floor((pos.x-imageLlcorner.x)/imageSize.x*matrixDimX);
        	m=(int) floor((pos.y-imageLlcorner.y)/imageSize.y*matrixDimY);
        	if(m<matrixDimY && n<matrixDimX && m>=0 && n>=0){
                	pointMatrix[m][n].points[tmpCounter[m][n]++] = points[i];
		}
	}
	for(int i=0; i<matrixDimY; i++){
		free(tmpCounter[i]);
	}
	free(tmpCounter);

	vect2 cellPos, pointPos, tolerance, pointContrib;
	tolerance.x = (tolX+0.5)*imageSize.x/matrixDimX;
	tolerance.y = (tolY+0.5)*imageSize.y/matrixDimY;
	for(int m=0; m<matrixDimY; m++){
		cellPos.y = imageLlcorner.y+(m+0.5)*imageSize.y/matrixDimY;
		for(int n=0; n<matrixDimX; n++){
			cellPos.x = imageLlcorner.x+(n+0.5)*imageSize.x/matrixDimX;
			pointMatrix[m][n].extContrib = backgroundBeta(cellPos);
			for(int i=0; i<pointc; i++){ //TODO smarter way?
				pointPos = points[i].pos;
				if(fabs(pointPos.x-cellPos.x)<tolerance.x || fabs(pointPos.y-cellPos.y)<tolerance.y)
					continue;
				pointContrib = pointBeta(cellPos.x-pointPos.x, cellPos.y-pointPos.y);
				pointMatrix[m][n].extContrib.x += pointContrib.x; //TODO different masses
				pointMatrix[m][n].extContrib.y += pointContrib.y;
			}
		}
	}
}
