.PHONY: build

build: rayshooter pointGen

rayshooter: main.o kernel.o storage.o kernel.h storage.h
	gcc main.o kernel.o storage.o -o rayshooter -lm -lconfig -g -p -pg
pointGen: pointGen.o storage.o kernel.o
	gcc pointGen.o storage.o kernel.o -o pointGen -lm -g -p -pg
main.o: main.c kernel.h
	gcc -std=c99 -c main.c -O3 -o main.o -g -p -pg
kernel.o: kernel.c kernel.h
	gcc -std=c99 -c kernel.c -O3 -o kernel.o -g -p -pg
storage.o: storage.c kernel.h storage.h
	gcc -std=c99 -c storage.c -O3 -o storage.o -g -p -pg
pointGen.o: pointGen.c storage.h kernel.h
	gcc -std=c99 -c pointGen.c -O3 -o pointGen.o -g -p -pg
