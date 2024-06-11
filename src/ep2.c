#define JMATRIX_IMPLEMENTATION
#define JNETWORK_IMPLEMENTATION
//#define JMATRIX_PRECISION long double
#include <jmatrix.h>
#include <jnetwork.h>
#include <time.h>

int main(int argc, char **argv) {
    srand(time(NULL));

    int layout[] = {2, 1};
    NN n = nn_alloc(2, 2, layout);
    nn_rand(n, 0, 1);
    NN_PRINT(n);

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            printf("Inputs: {%d, %d}\n", i, j);
            JMATRIX_PRECISION in[] = {i, j};
            Mat mIn = TO_1D_MAT(in, 2);
            NN_FORWARD_SIG(n, mIn);
            Mat out = NN_GET_OUT(n);
            MAT_PRINT(out);
        }
    }
    JMATRIX_PRECISION inp[][2] = {{0, 0}, {0 ,1}, {1, 0}, {1, 1}};
    Mat m_in = TO_2D_MAT(inp[0], 4, 2);
    JMATRIX_PRECISION outp[] = {0, 1, 1, 1};
    Mat m_out = TO_2D_MAT(outp, 4, 1);

    printf("Cost: %20.10Lf\n", (long double) NN_GET_COST_SIG(n, m_in, m_out, 4));

    return 0;
}