#define JMATRIX_IMPLEMENTATION
#define JNETWORK_IMPLEMENTATION
//#define JMATRIX_PRECISION long double
#include <jmatrix.h>
#include <jnetwork.h>
#include <time.h>

#define NEURONS 3
#define ARG_COUNT 5

int main(int argc, char **argv) {
    if (argc != ARG_COUNT + 1) {
        printf("Usage: %s <data> <debug> <rate> <eps> <iter>\n", argv[0]);
        return 1;
    }

    FILE *data = fopen(argv[1], "r");
    if (data == NULL) {
        printf("Error: could not open file %s\n", argv[1]);
        return 2;
    }

    Mat *test_data = nn_datafile_alloc(data);
    if (test_data == NULL) {
        printf("Error: loading from file: %s\n", argv[1]);
        return 7;
    }
    Mat in = test_data[0];
    Mat out = test_data[1];

    int debug;
    if (sscanf(argv[2], "%d", &debug) == EOF) {
        printf("Error: argument %s invalid\n", argv[2]);
        fclose(data);
        return 3;
    }

    float rate;
    if (sscanf(argv[3], "%f", &rate) == EOF) {
        printf("Error: argument %s invalid\n", argv[3]);
        fclose(data);
        return 4;
    }

    float eps;
    if (sscanf(argv[4], "%f", &eps) == EOF) {
        printf("Error: argument %s invalid\n", argv[4]);
        fclose(data);
        return 5;
    }

    size_t iterations;
    if (sscanf(argv[5], "%zu", &iterations) == EOF) {
        printf("Error: argument %s invalid\n", argv[5]);
        fclose(data);
        return 6;
    }

    srand(time(NULL));

    int layout[] = {2, 1};
    NN n;
    JMATRIX_PRECISION mem[NEURONS * 3];
    Mat mem_mat = {.mat = mem, .rows = 1, .cols = 9, .stride = 9};
    JMATRIX_PRECISION outMem[NEURONS];
    Mat outMem_mat = {.mat = outMem, .rows = 1, .cols = 3, .stride = 3};
    nn_init(&n, JMATRIX_MALLOC(sizeof(Mat) * NEURONS * 2), &mem_mat, &outMem_mat, 2, 2, layout);
    nn_fill(n, 1);

    JMATRIX_PRECISION test[] = {
        0, 0, 0,
        0, 1, 1,
        1, 0, 1,
        1, 1, 1
    };

    printf("Cost: %20.10Lf\n", (long double) NN_GET_COST_SIG(n, in, out));
    NN g = nn_alloc(2, 2, layout);
    #if 1
    for (int i = 0; i < iterations; i++) {
        nn_finite_diff(g, n, in, out, sigmoidP, eps);
        nn_learn(n, g, rate);
        if (debug != 0 && i % debug == 0) printf("Cost: %20.10Lf\n", (long double) NN_GET_COST_SIG(n, in, out));
    }
    #endif

    nn_finite_diff(g, n, in, out, sigmoidP, eps);
    nn_learn(n, g, rate);
    printf("Cost: %20.10Lf\n", (long double) NN_GET_COST_SIG(n, in, out));
    
    for (int i = 0; i < in.rows; i++) {
        Mat inputs = MAT_ROW(in, i);
        MAT_PRINT(inputs);
        NN_FORWARD_SIG(n, inputs);
        Mat result = NN_GET_OUT(n);
        MAT_PRINT(result);
    }

    //MAT_PRINT(outMem_mat);

    fclose(data);
    return 0;
}