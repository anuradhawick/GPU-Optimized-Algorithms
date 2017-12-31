package matrix;

import com.aparapi.Kernel;
import com.aparapi.Range;

/**
 * Created by anuradhawick on 12/31/17.
 * <p>
 * Note: Only supports square matrices. MxN matrix would require a small adjustment taking SIZE into two portions
 * row Size and col Size
 */
@SuppressWarnings("Duplicates")
public class MatrixTranspose {
    public static void main(String[] args) {
        // Width of the matrix
        final int SIZE = 12000;

        // We should use linear arrays as supported by the API
        final int[] a = new int[SIZE * SIZE];
        int[] c = new int[SIZE * SIZE];
        final int[] d = new int[SIZE * SIZE];
        int val;

        // Creating random matrices
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                a[i * SIZE + j] = (int) (Math.random() * 100);
            }
        }
        long time = System.currentTimeMillis();

        // CPU Transpose
        System.out.println("Starting single threaded computation");
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                writeMatrix(i, j, SIZE, c, readMatrix(j, i, SIZE, a));
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

                if (row > SIZE || col > SIZE) return;

                writeMatrix(row, col, SIZE, d, readMatrix(col, row, SIZE, a));
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
