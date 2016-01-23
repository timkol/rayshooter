#include <math.h>
#include <stdlib.h>
#include "kernel.h"

//#define backgroundBeta(imgPoint,dir) (0) //TODO beta from NFW
//#define pointBeta(imgX,imgY,dir) (0) //TODO beta from point mass

vect2 backgroundBeta(vect2 img) {
	decimal Tc=1;
	decimal e=0.2;
	decimal K=5;
	decimal F = (Tc*Tc+(1.-e)*img.x*img.x+(1.+e)*img.y*img.y);
	decimal tmp = K/sqrt(F);
	vect2 result;
	result.x = img.x*(1.-e)*tmp;
	result.y = img.y*(1.+e)*tmp;
	return result;
}

vect2 pointBeta(decimal imgX, decimal imgY) {
	decimal K=1;
	decimal tmp = K/(imgX*imgX+imgY*imgY);
	vect2 result;
	result.x = tmp*imgX;
	result.y = tmp*imgY;
	return result;
}
