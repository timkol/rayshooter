#include <stdlib.h>

#include "storage.h"

int main(){
	vect2 imgSize, imgLlcorner;
	imgSize.x = 20.0;
	imgSize.y = 20.0;
	imgLlcorner.x = -10.0;
	imgLlcorner.y = -10.0;
	createNew(NULL, 0, imgSize, imgLlcorner);
	serialize("cells.dat");
	freeMemory();
}
