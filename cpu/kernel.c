#include <math.h>
#include <stdlib.h>
#include "kernel.h"

//extern int pointc;
//extern vect2 *points;
//extern vect2 imageSize, imageLlcorner;

//#define backgroundBeta(imgPoint,dir) (0) //TODO beta from NFW
//#define pointBeta(imgX,imgY,dir) (0) //TODO beta from point mass

decimal backgroundBetaX(vect2 img) {
	decimal Tc=1;
	decimal e=0.2;
	decimal K=5;
	decimal F = (Tc*Tc+(1.-e)*img.x*img.x+(1.+e)*img.y*img.y);
	return img.x*K*(1.-e)/sqrt(F);
}

decimal backgroundBetaY(vect2 img) {
	decimal Tc=1;
	decimal e=0.2;
	decimal K=5;
	decimal F = (Tc*Tc+(1.-e)*img.x*img.x+(1.+e)*img.y*img.y);
	return img.y*K*(1.+e)/sqrt(F);
}

decimal pointBetaX(decimal imgX, decimal imgY) {
	decimal K=1;
	return K*imgX/(imgX*imgX+imgY*imgY);
}

decimal pointBetaY(decimal imgX, decimal imgY) {
	decimal K=1;
	return K*imgY/(imgX*imgX+imgY*imgY);
}
