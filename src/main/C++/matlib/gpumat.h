__device__ int readGMat(Matrix mat, int row, int col) {
	return mat.elements[row * mat.width + col];
}

__device__ void writeGMat(Matrix mat, int row, int col, int val) {
	mat.elements[row * mat.width + col] = val;
}
