#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

typedef struct {
	int width;
	int height;
	int* elements;
} Matrix;

#include "matrix_multiplication.h"

int main(int argc, char* argv[]) {
	// Size of vectors
	int SIZE = 1024;
	size_t size = SIZE * SIZE * sizeof(int);
	struct timeval start, end;

	// =====================================
	Matrix h_A;
	h_A.width = SIZE;
	h_A.height = SIZE;
	h_A.elements = (int*) malloc(size);

	Matrix d_A;
	d_A.width = SIZE;
	d_A.height = SIZE;

	// =====================================

	Matrix h_B;
	h_B.width = SIZE;
	h_B.height = SIZE;
	h_B.elements = (int*) malloc(size);

	Matrix d_B;
	d_B.width = SIZE;
	d_B.height = SIZE;
	d_B.elements = (int*) malloc(size);

	// =====================================

	Matrix h_C;
	h_C.width = SIZE;
	h_C.height = SIZE;
	h_C.elements = (int*) malloc(size);

	// Answer from CUDA
	Matrix h_dC;
	h_dC.width = SIZE;
	h_dC.height = SIZE;
	h_dC.elements = (int*) malloc(size);

	// Answer from CPU
	Matrix d_C;
	d_C.width = SIZE;
	d_C.height = SIZE;
	d_C.elements = (int*) malloc(size);

	// Initialize vectors on host
	for (int i = 0; i < SIZE; i++) {
		for (int j = 0; j < SIZE; ++j) {
			h_A.elements[i * SIZE + j] = rand() % 10;
			h_B.elements[i * SIZE + j] = rand() % 10;
		}
	}

	// Perform CPU calculation
	gettimeofday(&start, NULL);
	matMulCPU(h_A, h_B, h_C, SIZE);
	gettimeofday(&end, NULL);

	printf("CPU calculation ended in %ld\n",
			(end.tv_sec - start.tv_sec) * 1000
					+ (end.tv_usec - start.tv_usec) / 1000);

	// Copy host vectors to device
	gettimeofday(&start, NULL);

	cudaMalloc(&d_A.elements, size);
	cudaMalloc(&d_B.elements, size);
	cudaMalloc(&d_C.elements, size);

	cudaMemcpy(d_A.elements, h_A.elements, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B.elements, h_B.elements, size, cudaMemcpyHostToDevice);

	// Number of threads in each thread block
	dim3 threadsPerBlock(32, 32);

	// Number of thread blocks in grid
	dim3 numBlocks(SIZE / threadsPerBlock.x, SIZE / threadsPerBlock.y);

	// Execute the kernel
	matMulGPU<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, SIZE);
	cudaDeviceSynchronize();

	// Copy array back to host
	cudaMemcpy(h_dC.elements, d_C.elements, size, cudaMemcpyDeviceToHost);

	// Release device memory
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);

	gettimeofday(&end, NULL);

	printf("GPU calculation ended in %ld\n",
			(end.tv_sec - start.tv_sec) * 1000
					+ (end.tv_usec - start.tv_usec) / 1000);

	bool error = false;
	for (int i = 0; i < SIZE; ++i) {
		for (int j = 0; j < SIZE; ++j) {
			if (h_dC.elements[i * SIZE + j] != h_C.elements[i * SIZE + j]) {
				printf("ERROR %d %d\n", h_dC.elements[i * SIZE + j],
						h_C.elements[i * SIZE + j]);
				error = true;
				break;
			}

		}
		if (error) {
			break;
		}
	}

	// Release host memory
	free(h_A.elements);
	free(h_B.elements);
	free(h_C.elements);

	return 0;
}
