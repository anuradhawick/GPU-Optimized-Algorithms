#include "../matlib/gpumat.h"

// CUDA kernel.
__global__ void matMulGPU(Matrix A, Matrix B, Matrix C, int N) {
	// Get our global thread IDs
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// Make sure we do not go out of bounds
	if (row > N || col > N)
		return;
	int cVal = 0;
	for (int i = 0; i < N; ++i) {
		cVal += readGMat(A, row, i) * readGMat(B, i, col);
	}
	writeGMat(C, row, col, cVal);
}

void matMulCPU(Matrix A, Matrix B, Matrix C, int N) {
	int cVal = 0;
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			cVal = 0;
			for (int k = 0; k < N; ++k) {
				cVal += readMatrix(A, i, k) * readMatrix(B, k, j);
			}
			writeMatrix(C, i, j, cVal);
		}
	}
}
