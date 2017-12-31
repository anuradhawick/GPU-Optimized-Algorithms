package graph;


import com.aparapi.Kernel;
import com.aparapi.Range;

/**
 * Created by anuradhawick on 12/31/17.
 */
public class FloydWarshall {
    public static void main(String[] args) {
        final int SIZE = 1000;
        final int[] matrix = new int[SIZE * SIZE];
//        final int[] matrix = new int[]
//                {
//                        0, Integer.MAX_VALUE / 2, -2, Integer.MAX_VALUE / 2,
//                        4, 0, 3, Integer.MAX_VALUE / 2,
//                        Integer.MAX_VALUE / 2, Integer.MAX_VALUE / 2, 0, 2,
//                        Integer.MAX_VALUE / 2, -1, Integer.MAX_VALUE / 2, 2
//                };

        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                writeMatrix(i, j, SIZE, matrix, (int) (Math.random() * 10));
            }
        }
        final int[] cpuDistMatrix = new int[SIZE * SIZE];
        final int[] gpuDistMatrix = new int[SIZE * SIZE];

        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                if (i == j) {
                    writeMatrix(i, j, SIZE, cpuDistMatrix, 0);
                    writeMatrix(i, j, SIZE, gpuDistMatrix, 0);
                } else {
                    writeMatrix(i, j, SIZE, cpuDistMatrix, readMatrix(i, j, SIZE, matrix));
                    writeMatrix(i, j, SIZE, gpuDistMatrix, readMatrix(i, j, SIZE, matrix));
                }
            }
        }

        System.out.println("Starting CPU computation");
        long time = System.currentTimeMillis();
        for (int k = 0; k < SIZE; k++) {
            for (int i = 0; i < SIZE; i++) {
                for (int j = 0; j < SIZE; j++) {
                    if (readMatrix(i, j, SIZE, cpuDistMatrix) > readMatrix(i, k, SIZE, cpuDistMatrix) + readMatrix(k, j, SIZE, cpuDistMatrix)) {
                        writeMatrix(i, j, SIZE, cpuDistMatrix, readMatrix(i, k, SIZE, cpuDistMatrix) + readMatrix(k, j, SIZE, cpuDistMatrix));
                    }
                }
            }
        }
        System.out.println("Finished CPU computation " + (System.currentTimeMillis() - time));

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
                int i = getGlobalId() / SIZE;
                int j = getGlobalId() % SIZE;
                int k = getPassId();
                int dist = readMatrix(i, j, SIZE, gpuDistMatrix);
                int newDist = readMatrix(i, k, SIZE, gpuDistMatrix) + readMatrix(k, j, SIZE, gpuDistMatrix);

                if (i > SIZE || j > SIZE) return;

                if (dist > newDist) {
                    writeMatrix(i, j, SIZE, gpuDistMatrix, newDist);
                }
            }
        };


//        System.out.println("CPU Result");
//        for (int i = 0; i < SIZE; i++) {
//            for (int j = 0; j < SIZE; j++) {
//                System.out.print(readMatrix(i, j, SIZE, cpuDistMatrix) + "\t");
//            }
//            System.out.println("");
//        }

        System.out.println("Starting GPU computation");
        time = System.currentTimeMillis();
        // Array size for GPU to know
        Range range = Range.create(SIZE * SIZE);
        // Running the Kernel
        kernel.execute(range, SIZE);
        System.out.println("Finished GPU computation " + (System.currentTimeMillis() - time));

//        System.out.println("");
//        System.out.println("GPU Result");
//        for (int i = 0; i < SIZE; i++) {
//            for (int j = 0; j < SIZE; j++) {
//                System.out.print(readMatrix(i, j, SIZE, gpuDistMatrix) + "\t");
//            }
//            System.out.println("");
//        }

        // validation
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                if (readMatrix(i, j, SIZE, cpuDistMatrix) != readMatrix(i, j, SIZE, gpuDistMatrix)) {
                    System.out.println("ERROR");
                    System.out.println(readMatrix(i, j, SIZE, cpuDistMatrix) + " " + readMatrix(i, j, SIZE, gpuDistMatrix));
                    System.exit(1);
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
