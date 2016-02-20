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
vect2 borderedImageSize, borderedImageLlcorner;

void allocateMemory();

void serialize(const char* filename) {
	FILE *f;
	f = fopen(filename, "w");
	
	fprintf(f, "%x %x %x %x %a %a %a %a\n", matrixDimX, matrixDimY, tolX, tolY, borderedImageSize.x, borderedImageSize.y, borderedImageLlcorner.x, borderedImageLlcorner.y);

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
	
	fscanf(f, "%x %x %x %x %a %a %a %a", &matrixDimX, &matrixDimY, &tolX, &tolY, &borderedImageSize.x, &borderedImageSize.y, &borderedImageLlcorner.x, &borderedImageLlcorner.y);

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
	fprintf(stderr, "%f %f\n", borderedImageSize.x, borderedImageSize.y);
}

vect2 totalBeta(vect2 imgPos) {//TODO interpolation and near points
	int n=(int) floor((imgPos.x-borderedImageLlcorner.x)/borderedImageSize.x*matrixDimX);
	int m=(int) floor((imgPos.y-borderedImageLlcorner.y)/borderedImageSize.y*matrixDimY); //TODO rand() can return 1.0
/*	if(m>=matrixDimY || n>=matrixDimX || m<0 || n<0){
		fprintf(stderr, "%d %d %f %f\n", m, n, imgPos.x, imgPos.y);
		exit;
	}
*/
	vect2 beta = pointMatrix[m][n].extContrib;
        vect2 tmpBeta;
        cell thisCell;
        for(int j=n-tolX; j<=n+tolX; j++){
            for(int i=m-tolY; i<=m+tolY; i++){
                thisCell = pointMatrix[i][j];
                for(int k=0;k<thisCell.pointc; k++){
                    tmpBeta = pointBeta(imgPos.x-thisCell.points[k].pos.x, imgPos.y-thisCell.points[k].pos.y);//TODO different point masses
                    beta.x += tmpBeta.x;
                    beta.y += tmpBeta.y;
                }
            }
        }
        return beta;
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

void setParams(vect2 imgSize, vect2 imgLlcorner) {
    /* boundary around original shooting area in cell count "units"
     * r & u margin should be l & d +1, because rand()/RAND_MAX can return 1 
     * and they (l & d) should be at least equal to tolX & tolY +1 (rounding errors) */
    int lmargin = 11;
    int rmargin = 12;
    int umargin = 12;
    int dmargin = 11;
    
    int origMatrixDimX=1000; //TODO user defined values
    int origMatrixDimY=1000;
    tolX=10;
    tolY=10;
    
    vect2 cellSize;
    cellSize.x = imgSize.x/origMatrixDimX;
    cellSize.y = imgSize.y/origMatrixDimY;
    
    borderedImageSize.x = imgSize.x + (lmargin+rmargin)*cellSize.x;
    borderedImageSize.y = imgSize.y + (umargin+dmargin)*cellSize.y;
    
    borderedImageLlcorner.x = imgLlcorner.x-lmargin*cellSize.x;
    borderedImageLlcorner.y = imgLlcorner.y-dmargin*cellSize.y;
    
    matrixDimX = origMatrixDimX + lmargin + rmargin;
    matrixDimY = origMatrixDimY + umargin + dmargin;
}

void createNew(pointMass* points, int pointc, vect2 imgSize, vect2 imgLlcorner) {
	
        setParams(imgSize, imgLlcorner);
	allocateMemory();

	//populate
	int n,m;
	vect2 pos;
	for(int i=0; i<pointc; i++){
		pos = points[i].pos;
        	n=(int) floor((pos.x-borderedImageLlcorner.x)/borderedImageSize.x*matrixDimX);
        	m=(int) floor((pos.y-borderedImageLlcorner.y)/borderedImageSize.y*matrixDimY);
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
        	n=(int) floor((pos.x-borderedImageLlcorner.x)/borderedImageSize.x*matrixDimX);
        	m=(int) floor((pos.y-borderedImageLlcorner.y)/borderedImageSize.y*matrixDimY);
        	if(m<matrixDimY && n<matrixDimX && m>=0 && n>=0){
                	pointMatrix[m][n].points[tmpCounter[m][n]++] = points[i];
		}
	}
	for(int i=0; i<matrixDimY; i++){
		free(tmpCounter[i]);
	}
	free(tmpCounter);

	vect2 cellPos, pointPos, tolerance, pointContrib;
	tolerance.x = (tolX+0.5)*borderedImageSize.x/matrixDimX;
	tolerance.y = (tolY+0.5)*borderedImageSize.y/matrixDimY;
	for(int m=0; m<matrixDimY; m++){
		cellPos.y = borderedImageLlcorner.y+(m+0.5)*borderedImageSize.y/matrixDimY;
		for(int n=0; n<matrixDimX; n++){
			cellPos.x = borderedImageLlcorner.x+(n+0.5)*borderedImageSize.x/matrixDimX;
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
