#ifndef JNETWORK_H
#define JNETWORK_H

#include <jmatrix.h>
#include <stdbool.h>

typedef struct {
    Mat *w, *b, *out;
    int size;
} NN;

#define NN_FORWARD_SIG(n, in) nn_forward(n, in, sigmoidP)
#define NN_GET_COST_SIG(n, in, out) nn_get_cost(n, in, out, sigmoidP)
#define NN_GET_OUT(n) (n.out[n.size - 1])
#define NN_PRINT(n) nn_print(#n, n)

NN nn_alloc(int size, int inSize, int *lSizes);
void nn_init(NN *n, Mat *mem, Mat *paramMem, Mat *outMem, int size, int inSize, int *lSizes);
void nn_fill(NN n, JMATRIX_PRECISION val);
void nn_copy(NN dst, NN src, bool out);
void nn_rand(NN n, JMATRIX_PRECISION low, JMATRIX_PRECISION high);
void nn_forward(NN n, Mat in, JMATRIX_PRECISION (*f)(JMATRIX_PRECISION));
JMATRIX_PRECISION nn_get_cost(NN n, Mat in, Mat out, JMATRIX_PRECISION (*f)(JMATRIX_PRECISION));
void nn_finite_diff(NN g, NN n, Mat in, Mat out, JMATRIX_PRECISION (*f)(JMATRIX_PRECISION), JMATRIX_PRECISION eps);
void nn_learn(NN out, NN g, JMATRIX_PRECISION rate);
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
    mat_init(n.w, inSize, lSizes[0], lSizes[0]);
    mat_init(n.b, 1, lSizes[0], lSizes[0]);
    mat_init(n.out, 1, lSizes[0], lSizes[0]);
    for (int i = 1; i < size; i++) {
        mat_init(&n.w[i], lSizes[i - 1], lSizes[i], lSizes[i]);
        mat_init(&n.b[i], 1, lSizes[i], lSizes[i]);
        mat_init(&n.out[i], 1, lSizes[i], lSizes[i]);
    }
    return n;
}

void nn_init(NN *n, Mat *mem, Mat *paramMem, Mat *outMem, int size, int inSize, int *lSizes) {
    int memSize = 0;
    for (int i = 0; i < size; i++) memSize += lSizes[i];
    n->w = &mem[0];
    n->b = &mem[sizeof(Mat) * memSize];
    n->out = &mem[sizeof(Mat) * memSize * 2];
    n->size = size;
    n->w[0] = MAT_SUB_AT(*paramMem, inSize, lSizes[0], lSizes[0], 0, 0);
    n->b[0] = MAT_SUB_AT(n->w[0], 1, lSizes[0], lSizes[0], inSize, 0);
    n->out[0] = MAT_SUB_AT(*outMem, 1, lSizes[0], lSizes[0], 0, 0);
    for (int i = 1; i < size; i++) {
        n->w[i] = MAT_SUB_AT(n->b[i - 1], lSizes[i - 1], lSizes[i], lSizes[i], 1, 0);
        n->b[i] = MAT_SUB_AT(n->w[i], 1, lSizes[i], lSizes[i], lSizes[i - 1], 0);
        n->out[i] = MAT_SUB_AT(n->out[i - 1], 1, lSizes[i], lSizes[i], 1, 0);
    }
}

void nn_fill(NN n, JMATRIX_PRECISION val) {
    for (int i = 0; i < n.size; i++) {
        mat_fill(n.w[i], val);
        mat_fill(n.b[i], val);
        mat_fill(n.out[i], val);
    }
}

void nn_copy(NN dst, NN src, bool out) {
    JMATRIX_ASSERT(dst.size = src.size);
    for (int i = 0; i < src.size; i++) {
        mat_copy(dst.w[i], src.w[i]);
        mat_copy(dst.b[i], src.b[i]);
        if (out) mat_copy(dst.out[i], src.out[i]);
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

JMATRIX_PRECISION nn_get_cost(NN n, Mat in, Mat out, JMATRIX_PRECISION (*f)(JMATRIX_PRECISION)) {
    JMATRIX_ASSERT(in.cols == n.b[0].cols);
    JMATRIX_ASSERT(in.rows == out.rows);
    JMATRIX_ASSERT(out.cols == NN_GET_OUT(n).cols);

    JMATRIX_PRECISION result = 0;

    for (int i = 0; i < in.rows; i++) {
        nn_forward(n, MAT_ROW(in, i), f);
        JMATRIX_PRECISION dTotal = 0;
        for (int j = 0; j < out.cols; j++) {
            JMATRIX_PRECISION diff = MAT_AT(NN_GET_OUT(n), i, j) - MAT_AT(out, i, j);
            dTotal += diff * diff;
        }
        result += dTotal / out.cols;
    }

    return result / in.rows;
}

void nn_finite_diff(NN g, NN n, Mat in, Mat out, JMATRIX_PRECISION (*f)(JMATRIX_PRECISION), JMATRIX_PRECISION eps) {
    JMATRIX_ASSERT(g.size == n.size);
    JMATRIX_PRECISION buffer;

    JMATRIX_PRECISION cost = nn_get_cost(n, in, out, f);

    for (int i = 0; i < n.size; i++) {
        for (int j = 0; j < n.w[i].cols; j++) {
            for (int k = 0; k < n.w[i].rows; k++) {
                buffer = MAT_AT(n.w[i], k, j);
                MAT_AT(n.w[i], k, j) += eps;
                MAT_AT(g.w[i], k, j) = (nn_get_cost(n, in, out, f) - cost) / eps;
                MAT_AT(n.w[i], k, j) = buffer;
            }
            buffer = MAT_AT(n.b[i], 0, j);
            MAT_AT(n.b[i], 0, j) += eps;
            MAT_AT(g.b[i], 0, j) = (nn_get_cost(n, in, out, f) - cost) / eps;
            MAT_AT(n.b[i], 0, j) = buffer;
        }
    }
}

void nn_learn(NN out, NN g, JMATRIX_PRECISION rate) {
    JMATRIX_ASSERT(g.size == out.size);

    for (int i = 0; i < out.size; i++) {
        for (int j = 0; j < out.w[i].cols; j++) {
            for (int k = 0; k < out.w[i].rows; k++) {
                MAT_AT(out.w[i], k, j) -= rate * MAT_AT(g.w[i], k, j);
            }
            MAT_AT(out.b[i], 0, j) -= rate * MAT_AT(g.b[i], 0, j);
        }
    }
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