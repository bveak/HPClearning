CC = nvcc

task1: main.o add.o
	$(CC) main.o add.o -o task1 -O2
main.o: main.cu add.cuh
	$(CC) main.cu -c main.o -O2
add.o: add.cu add.cuh
	$(CC) add.cu -c add.o -O2

.PHONY: clean
clean:
	-rm main main.o add.o