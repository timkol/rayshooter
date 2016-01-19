rayshooter: main.o kernel.o kernel.h
	gcc main.o kernel.o -o rayshooter -lm -lconfig -g -p -pg
main.o: main.c kernel.h
	gcc -std=c99 -c main.c  -O3 -o main.o -g -p -pg
kernel.o: kernel.c kernel.h
	gcc -std=c99 -c kernel.c -O3 -o kernel.o -g -p -pg
