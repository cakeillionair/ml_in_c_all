#define JMATRIX_IMPLEMENTATION
#define JNETWORK_IMPLEMENTATION
//#define JMATRIX_PRECISION long double
#include <jmatrix.h>
#include <jnetwork.h>
#include <time.h>

void printArr(JMATRIX_PRECISION *arr, int size) {
    printf("{ ");
    for (int i = 0; i < size; i++) printf("%Lf ", (long double) arr[i]);
    printf("}\n");
}

int main(int argc, char **argv) {
    srand(time(NULL));

    int layout[] = {2, 1};
    NN n;
    JMATRIX_PRECISION mem[12];
    Mat mem_mat = {.mat = mem, .rows = 0, .cols = 0, .stride = 0};
    nn_init(&n, JMATRIX_MALLOC(sizeof(Mat) * 3 * 3), &mem_mat, 2, 2, layout);
    nn_rand(n, 0, 1);
    NN_PRINT(n);

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            printf("Inputs: {%d, %d}\n", i, j);
            JMATRIX_PRECISION in[] = {i, j};
            NN_FORWARD_SIG(n, TO_1D_MAT(in, 2));
            Mat out = NN_GET_OUT(n);
            MAT_PRINT(out);
        }
    }
    JMATRIX_PRECISION inp[][2] = {{0, 0}, {0 ,1}, {1, 0}, {1, 1}};
    JMATRIX_PRECISION outp[] = {0, 1, 1, 1};
    JMATRIX_PRECISION test[] = {
        0, 0, 0,
        0, 1, 1,
        1, 0, 1,
        1, 1, 1
    };

    printf("Cost: %20.10Lf\n", (long double) NN_GET_COST_SIG(n, TO_2D_MAT(test, 4, 2, 3), TO_2D_MAT(test, 4, 1, 3), 4));

    printArr(mem_mat.mat, 12);

    return 0;
}