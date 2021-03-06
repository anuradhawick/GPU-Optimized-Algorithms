package matrix;

import com.aparapi.Kernel;
import com.aparapi.Range;

/**
 * Created by anuradhawick on 12/29/17.
 */
@SuppressWarnings("Duplicates")
public class MatrixMultiplication {

    public static void main(String[] args) {
        // Width of the matrix
        final int SIZE = 1000;

        // We should use linear arrays as supported by the API
        final int[] a = new int[SIZE * SIZE];
        final int[] b = new int[SIZE * SIZE];
        int[] c = new int[SIZE * SIZE];
        final int[] d = new int[SIZE * SIZE];
        int val;

        // Creating random matrices
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                a[i * SIZE + j] = (int) (Math.random() * 100);
                b[i * SIZE + j] = (int) (Math.random() * 100);
            }
        }
        long time = System.currentTimeMillis();

        // CPU multiplication
        System.out.println("Starting single threaded computation");
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                val = 0;
                for (int k = 0; k < SIZE; k++) {
                    val += readMatrix(i, k, SIZE, a) * readMatrix(k, j, SIZE, b);
                }
                writeMatrix(i, j, SIZE, c, val);
            }
        }
        System.out.println("Task finished in " + (System.currentTimeMillis() - time) + "ms");

        // Kernel for multiplication
        Kernel kernel = new Kernel() {
            int readMatrix(int i, int j, int rs, int[] matrix) {
                return matrix[i * rs + j];
            }

            int[] writeMatrix(int i, int j, int rs, int[] matrix, int val) {
                matrix[i * rs + j] = val;
                return matrix;
            }

            @Override
            public void run() {
                int row = getGlobalId() / SIZE;
                int col = getGlobalId() % SIZE;
                int temp;

                if (row > SIZE || col > SIZE) return;

                writeMatrix(row, col, SIZE, d, 0);

                for (int i = 0; i < SIZE; i++) {
                    temp = readMatrix(row, col, SIZE, d) + readMatrix(row, i, SIZE, a) * readMatrix(i, col, SIZE, b);
                    writeMatrix(row, col, SIZE, d, temp);
                }
            }
        };

        // Array size for GPU to know
        Range range = Range.create(SIZE * SIZE);

        System.out.println("Starting GPU computation");
        time = System.currentTimeMillis();
        kernel.execute(range); // Running the Kernel
        System.out.println("Task finished in " + (System.currentTimeMillis() - time) + "ms");

        // Verifying the result
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                if (c[i * SIZE + j] != d[i * SIZE + j]) {
                    System.out.println("ERROR");
                    return;
                }
            }
        }
    }

    static int readMatrix(int i, int j, int rs, int[] matrix) {
        return matrix[i * rs + j];
    }

    static int[] writeMatrix(int i, int j, int rs, int[] matrix, int val) {
        matrix[i * rs + j] = val;
        return matrix;
    }
}
