typedef struct {
	int width;
	int height;
	int* elements;
} Matrix;

Matrix mallocMatrix(int width, int height);
Matrix getRandomMatrix(int width, int height);
int readMatrix(Matrix mat, int row, int col);
void writeMatrix(Matrix mat, int row, int col, int val);
