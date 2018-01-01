#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#include "../../matlib/matrix.h"
#include "floydWarshall.h"

int main() {
	// Size of vectors
	int SIZE = 320;
	size_t size = SIZE * SIZE * sizeof(int);

	struct timeval start, end;

	// =====================================
	Matrix dist = getRandomMatrix(SIZE, SIZE);
	Matrix distGPU = mallocMatrix(SIZE, SIZE);

	for (int i = 0; i < SIZE; ++i) {
		writeMatrix(dist, i, i, 0);
	}

	// obtaining a copy for GPU computing
	for (int i = 0; i < SIZE; ++i) {
		for (int j = 0; j < SIZE; ++j) {
			writeMatrix(distGPU, i, j, readMatrix(dist, i, j));
		}
	}

	// Perform CPU calculation
	gettimeofday(&start, NULL);
	floydCPU(dist);
	gettimeofday(&end, NULL);
	printf("CPU calculation ended in %ld\n",
			(end.tv_sec - start.tv_sec) * 1000
					+ (end.tv_usec - start.tv_usec) / 1000);

	// =====================================
	Matrix d_distGPU;
	d_distGPU.width = SIZE;
	d_distGPU.height = SIZE;

	// Copy host vectors to device
	gettimeofday(&start, NULL);
	cudaMalloc(&d_distGPU.elements, size);
	cudaMemcpy(d_distGPU.elements, distGPU.elements, size,
			cudaMemcpyHostToDevice);

	// Number of threads in each thread block
	dim3 threadsPerBlock(32, 32);
	// Number of thread blocks in grid
	dim3 numBlocks(SIZE / threadsPerBlock.x, SIZE / threadsPerBlock.y);

	// Execute the kernel
	for (int k = 0; k < SIZE; ++k) {
		floydGPU<<<numBlocks, threadsPerBlock>>>(d_distGPU, k);
	}
	cudaDeviceSynchronize();

	// Copy array back to host
	cudaMemcpy(distGPU.elements, d_distGPU.elements, size,
			cudaMemcpyDeviceToHost);

	// Release device memory
	cudaFree(d_distGPU.elements);

	gettimeofday(&end, NULL);

	printf("GPU calculation ended in %ld\n",
			(end.tv_sec - start.tv_sec) * 1000
					+ (end.tv_usec - start.tv_usec) / 1000);

	bool error = false;
	for (int i = 0; i < SIZE; ++i) {
		for (int j = 0; j < SIZE; ++j) {
			if (dist.elements[i * SIZE + j] != distGPU.elements[i * SIZE + j]) {
				printf("ERROR %d %d\n", distGPU.elements[i * SIZE + j],
						dist.elements[i * SIZE + j]);
				error = true;
				break;
			}

		}
		if (error) {
			break;
		}
	}

	// Release host memory
	free(dist.elements);
	free(distGPU.elements);

	return 0;
}
