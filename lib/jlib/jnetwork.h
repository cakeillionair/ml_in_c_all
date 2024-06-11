#ifndef JNETWORK_H
#define JNETWORK_H

#include <jmatrix.h>

typedef struct {
    Mat *w, *b, *out;
    int size;
} NN;

#define NN_FORWARD_SIG(n, in) nn_forward(n, in, sigmoidP)
#define NN_GET_COST_SIG(n, in, out, amount) nn_get_cost(n, in, out, amount, sigmoidP)
#define NN_GET_OUT(n) (n.out[n.size - 1])
#define NN_PRINT(n) nn_print(#n, n)

NN nn_alloc(int size, int inSize, int *lSizes);
void nn_fill(NN n, JMATRIX_PRECISION val);
void nn_rand(NN n, JMATRIX_PRECISION low, JMATRIX_PRECISION high);
void nn_forward(NN n, Mat in, JMATRIX_PRECISION (*f)(JMATRIX_PRECISION));
JMATRIX_PRECISION nn_get_cost(NN n, Mat in, Mat out, int amount, JMATRIX_PRECISION (*f)(JMATRIX_PRECISION));
void nn_print(const char *name, NN n);

#endif // Header

#ifdef JNETWORK_IMPLEMENTATION
#undef JNETWORK_IMPLEMENTATION

NN nn_alloc(int size, int inSize, int *lSizes) {
    JMATRIX_ASSERT(size > 0);
    JMATRIX_ASSERT(inSize > 0);
    JMATRIX_ASSERT(lSizes != NULL || size == 1);
    NN n;
    n.size = size;
    int memSize = 0;
    for (int i = 0; i < size; i++) memSize += lSizes[i];
    n.w = JMATRIX_MALLOC(sizeof(Mat) * memSize);
    n.b = JMATRIX_MALLOC(sizeof(Mat) * memSize);
    n.out = JMATRIX_MALLOC(sizeof(Mat) * memSize);
    mat_init(n.w, inSize, lSizes[0]);
    mat_init(n.b, 1, lSizes[0]);
    mat_init(n.out, 1, lSizes[0]);
    for (int i = 1; i < size; i++) {
        mat_init(&n.w[i], inSize, lSizes[i]);
        mat_init(&n.b[i], 1, lSizes[i]);
        mat_init(&n.out[i], 1, lSizes[i]);
    }
    return n;
}

void nn_fill(NN n, JMATRIX_PRECISION val) {
    for (int i = 0; i < n.size; i++) {
        mat_fill(n.w[i], val);
        mat_fill(n.b[i], val);
        mat_fill(n.out[i], val);
    }
}

void nn_rand(NN n, JMATRIX_PRECISION low, JMATRIX_PRECISION high) {
    for (int i = 0; i < n.size; i++) {
        mat_rand(n.w[i], low, high);
        mat_rand(n.b[i], low, high);
        mat_rand(n.out[i], low, high);
    }
}

void nn_forward(NN n, Mat in, JMATRIX_PRECISION (*f)(JMATRIX_PRECISION)) {
    JMATRIX_ASSERT(n.size > 0);
    mat_dot(n.out[0], in, n.w[0]);
    mat_sum(n.out[0], n.b[0]);
    mat_apply(n.out[0], f);
    for (int i = 1; i < n.size; i++) {
        mat_dot(n.out[i], n.out[i - 1], n.w[i]);
        mat_sum(n.out[i], n.b[i]);
        mat_apply(n.out[i], f);
    }
}

JMATRIX_PRECISION nn_get_cost(NN n, Mat in, Mat out, int amount, JMATRIX_PRECISION (*f)(JMATRIX_PRECISION)) {
    JMATRIX_PRECISION result = 0;

    for (int i = 0; i < amount; i++) {
        Mat in_curr = TO_1D_MAT(&MAT_AT(in, i, 0), in.cols);
        nn_forward(n, in_curr, f);
        Mat act = NN_GET_OUT(n);
        JMATRIX_PRECISION dTotal = 0;
        for (int j = 0; j < out.cols; j++) {
            JMATRIX_PRECISION y = MAT_AT(act, 0, j);
            JMATRIX_PRECISION diff = y - MAT_AT(out, i, j);
            dTotal += diff * diff;
        }
        //printf("debug result: %Lf, dTotal: %Lf, cols: %d, amount: %d\n", (long double) result, (long double) dTotal, out.cols, amount);
        result += dTotal / out.cols;
    }

    return result / amount;
}

void nn_print(const char *name, NN n) {
    printf("Network: %s\n", name);
    for (int i = 0; i < n.size; i++) {
        printf("Layer %i:\n", i);
        mat_print("Weights", n.w[i]);
        mat_print("Biases", n.b[i]);
        mat_print("Outputs", n.out[i]);
    }
}

#endif // Implementation