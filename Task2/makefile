CC = nvcc

task1: main.o mul.o
	$(CC) main.o mul.o -o task2 -O2
main.o: main.cu mul.cuh
	$(CC) main.cu -c main.o -O2
mul.o: mul.cu mul.cuh
	$(CC) mul.cu -c mul.o -O2

.PHONY: clean
clean:
	-rm main main.o mul.o