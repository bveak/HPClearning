CC = nvcc

task1: main.o debubble.o
	$(CC) main.o debubble.o -o task4 -O2
main.o: main.cu debubble.cuh
	$(CC) main.cu -c main.o -O2
debubble.o: debubble.cu debubble.cuh
	$(CC) debubble.cu -c debubble.o -O2

.PHONY: clean
clean:
	-rm main main.o debubble.o