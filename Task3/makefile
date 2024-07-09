CC = nvcc

task1: main.o sort.o
	$(CC) main.o sort.o -o task3 -O2
main.o: main.cu sort.cuh
	$(CC) main.cu -c main.o -O2
sort.o: sort.cu sort.cuh
	$(CC) sort.cu -c sort.o -O2

.PHONY: clean
clean:
	-rm main main.o sort.o