#define JMATRIX_IMPLEMENTATION
#define JNETWORK_IMPLEMENTATION
//#define JMATRIX_PRECISION long double
#include <jmatrix.h>
#include <jnetwork.h>
#include <time.h>

#define NEURONS 3

int main(int argc, char **argv) {
    srand(time(NULL));
    float eps = 1e-1;
    float rate = 1e-1;

    int layout[] = {2, 1};
    NN n;
    JMATRIX_PRECISION mem[NEURONS * 3];
    Mat mem_mat = {.mat = mem, .rows = 1, .cols = 9, .stride = 9};
    JMATRIX_PRECISION outMem[NEURONS];
    Mat outMem_mat = {.mat = outMem, .rows = 1, .cols = 3, .stride = 3};
    nn_init(&n, JMATRIX_MALLOC(sizeof(Mat) * NEURONS * 2), &mem_mat, &outMem_mat, 2, 2, layout);
    nn_fill(n, 1);

    JMATRIX_PRECISION inp[][2] = {{0, 0}, {0 ,1}, {1, 0}, {1, 1}};
    JMATRIX_PRECISION outp[] = {0, 1, 1, 1};
    JMATRIX_PRECISION test[] = {
        0, 0, 0,
        0, 1, 1,
        1, 0, 1,
        1, 1, 1
    };

    printf("Cost: %20.10Lf\n", (long double) NN_GET_COST_SIG(n, TO_2D_MAT(test, 4, 2, 3), TO_2D_MAT(&test[2], 4, 1, 3)));
    NN g = nn_alloc(2, 2, layout);
    #if 1
    for (int i = 0; i < 1000000; i++) {
        nn_finite_diff(g, n, TO_2D_MAT(test, 4, 2, 3), TO_2D_MAT(&test[2], 4, 1, 3), sigmoidP, eps);
        nn_learn(n, g, rate);
        printf("Cost: %20.10Lf\n", (long double) NN_GET_COST_SIG(n, TO_2D_MAT(test, 4, 2, 3), TO_2D_MAT(&test[2], 4, 1, 3)));
    }
    #endif

    nn_finite_diff(g, n, TO_2D_MAT(test, 4, 2, 3), TO_2D_MAT(&test[2], 4, 1, 3), sigmoidP, eps);
    nn_learn(n, g, rate);
    printf("Cost: %20.10Lf\n", (long double) NN_GET_COST_SIG(n, TO_2D_MAT(test, 4, 2, 3), TO_2D_MAT(&test[2], 4, 1, 3)));
    
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            printf("Inputs: {%d, %d}\n", i, j);
            JMATRIX_PRECISION in[] = {i, j};
            NN_FORWARD_SIG(n, TO_1D_MAT(in, 2));
            printf("Result: %Lf\n", (long double) NN_GET_OUT(n).mat[0]);
        }
    }

    MAT_PRINT(mem_mat);
    MAT_PRINT(n.w[0]);
    MAT_PRINT(n.w[1]);
    MAT_PRINT(n.b[0]);
    MAT_PRINT(n.b[1]);
    MAT_PRINT(outMem_mat);
    MAT_PRINT(TO_2D_MAT(&test[2], 4, 1, 3));

    return 0;
}