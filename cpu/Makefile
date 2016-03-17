.PHONY: build

build: rayshooter pointGen

DBGFLAGS=-g -p -pg
#DBGFLAGS= 
#OPTFLAGS=-O3
OPTFLAGS=

rayshooter: main.o kernel.o storage.o kernel.h storage.h
	gcc main.o kernel.o storage.o -o rayshooter -lm -lconfig $(DBGFLAGS)
pointGen: pointGen.o storage.o kernel.o
	gcc pointGen.o storage.o kernel.o -o pointGen -lm $(DBGFLAGS)
main.o: main.c kernel.h
	gcc -std=c99 -c main.c -o main.o $(OPTFLAGS) $(DBGFLAGS)
kernel.o: kernel.c kernel.h
	gcc -std=c99 -c kernel.c -o kernel.o $(OPTFLAGS) $(DBGFLAGS)
storage.o: storage.c kernel.h storage.h
	gcc -std=c99 -c storage.c -o storage.o $(OPTFLAGS) $(DBGFLAGS)
pointGen.o: pointGen.c storage.h kernel.h
	gcc -std=c99 -c pointGen.c -o pointGen.o $(OPTFLAGS) $(DBGFLAGS)
