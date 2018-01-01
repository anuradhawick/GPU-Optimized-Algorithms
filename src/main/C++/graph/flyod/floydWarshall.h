#include "../../matlib/gpumat.h"

// CUDA kernel.
__global__ void floydGPU(Matrix dist, int k) {
	// Get our global thread IDs
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (dist.elements[i * dist.width + j]
			> dist.elements[i * dist.width + k]
					+ dist.elements[k * dist.width + j]) {
		dist.elements[i * dist.width + j] = dist.elements[i * dist.width + k]
				+ dist.elements[k * dist.width + j];
	}

}

void floydCPU(Matrix dist) {
	for (int k = 0; k < dist.width; ++k) {
		for (int i = 0; i < dist.width; ++i) {
			for (int j = 0; j < dist.width; ++j) {
				if (dist.elements[i * dist.width + j]
						> dist.elements[i * dist.width + k]
								+ dist.elements[k * dist.width + j]) {
					dist.elements[i * dist.width + j] = dist.elements[i
							* dist.width + k]
							+ dist.elements[k * dist.width + j];
				}
			}
		}
	}
}
