#include <cstdlib>

using namespace std;

#include "matrix.h"

Matrix mallocMatrix(int width, int height) {
	Matrix mat = Matrix();
	mat.width = width;
	mat.height = height;
	mat.elements = (int*) malloc(width * height * sizeof(int));
	return mat;
}

Matrix getRandomMatrix(int width, int height) {
	Matrix mat = mallocMatrix(width, height);
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			writeMatrix(mat, i, j, rand() % 10);
		}
	}

	return mat;
}

int readMatrix(Matrix mat, int row, int col) {
	return mat.elements[row * mat.width + col];
}

void writeMatrix(Matrix mat, int row, int col, int val) {
	mat.elements[row * mat.width + col] = val;
}
