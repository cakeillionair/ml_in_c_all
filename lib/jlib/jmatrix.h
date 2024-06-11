#ifndef JMATRIX_H
#define JMATRIX_H

#include <stdio.h>
#include <math.h>

#ifndef JMATRIX_MALLOC
#include <stdlib.h>
#define JMATRIX_MALLOC malloc
#endif

#ifndef JMATRIX_ASSERT
#include <assert.h>
#define JMATRIX_ASSERT assert
#endif

#ifndef JMATRIX_PRECISION
#define JMATRIX_PRECISION float
#endif

typedef struct {
    JMATRIX_PRECISION *mat;
    int rows;
    int cols;
} Mat;

#define TO_1D_MAT(arr, s) ((Mat) {.mat = arr, .rows = 1, .cols = s})
#define TO_2D_MAT(arr, r, c) ((Mat) {.mat = arr, .rows = r, .cols = c})
#define MAT_AT(m, i, j) (m).mat[(i) * (m.cols) + (j)]
#define MAT_ROW(m, i) TO_1D_MAT(&MAT_AT(m, i, 0), (m.cols))
#define MAT_PRINT(m) mat_print(#m, m)
#define MAT_SIG(m) mat_apply(m, sigmoidP);

JMATRIX_PRECISION rand_JMATRIX_PRECISION();
JMATRIX_PRECISION sigmoidP(JMATRIX_PRECISION x);

Mat mat_alloc(int rows, int cols);
void mat_init(Mat *m, int rows, int cols);
void mat_fill(Mat m, JMATRIX_PRECISION val);
void mat_rand(Mat m, JMATRIX_PRECISION low, JMATRIX_PRECISION high);
void mat_dot(Mat out, Mat a, Mat b);
void mat_sum(Mat out, Mat a);
void mat_apply(Mat m, JMATRIX_PRECISION (*f)(JMATRIX_PRECISION));
void mat_print(const char *name, Mat a);

#endif // Header

#ifdef JMATRIX_IMPLEMENTATION
#undef JMATRIX_IMPLEMENTATION

JMATRIX_PRECISION rand_JMATRIX_PRECISION() {
    return (JMATRIX_PRECISION) rand() / (JMATRIX_PRECISION) RAND_MAX;
}

JMATRIX_PRECISION sigmoidP(JMATRIX_PRECISION x) {
    return 1. / (1. + exp(x));
}

Mat mat_alloc(int rows, int cols) {
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.mat = JMATRIX_MALLOC(sizeof(JMATRIX_PRECISION) * rows * cols);
    JMATRIX_ASSERT(m.mat != NULL);
    return m;
}

void mat_init(Mat *m, int rows, int cols) {
    m->rows = rows;
    m->cols = cols;
    m->mat = JMATRIX_MALLOC(sizeof(JMATRIX_PRECISION) * rows * cols);
    JMATRIX_ASSERT(m->mat != NULL);
}

void mat_fill(Mat m, JMATRIX_PRECISION val) {
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            MAT_AT(m, i, j) = val;
        }
    }
}

void mat_copy(Mat dst, Mat src) {
    JMATRIX_ASSERT(dst.rows == src.rows);
    JMATRIX_ASSERT(dst.cols == src.cols);
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            MAT_AT(dst, i, j) = MAT_AT(src, i, j);
        }
    }
}

void mat_rand(Mat m, JMATRIX_PRECISION low, JMATRIX_PRECISION high) {
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            MAT_AT(m, i, j) = rand_JMATRIX_PRECISION() * (high - low) + low;
        }
    }
}

void mat_dot(Mat out, Mat a, Mat b) {
    JMATRIX_ASSERT(a.cols == b.rows);
    JMATRIX_ASSERT(out.rows == a.rows);
    JMATRIX_ASSERT(out.cols == b.cols);
    int n = a.cols;

    for (int i = 0; i < out.rows; i++) {
        for (int j = 0; j < out.cols; j++) {
            MAT_AT(out, i, j) = 0;
            for (int k = 0; k < n; k++) {
                MAT_AT(out, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
            }
        }
    }
}

void mat_sum(Mat out, Mat m) {
    JMATRIX_ASSERT(out.rows == m.rows);
    JMATRIX_ASSERT(out.cols == m.cols);
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            MAT_AT(out, i, j) += MAT_AT(m, i, j);
        }
    }
}

void mat_apply(Mat m, JMATRIX_PRECISION (*f)(JMATRIX_PRECISION)) {
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            MAT_AT(m, i, j) = (*f)(MAT_AT(m, i, j));
        }
    }
}

void mat_print(const char *name, Mat m) {
    if (name != NULL) printf("Matrix: %s\n", name);
    for (int i = 0; i < m.rows; i++) {
        printf("{ ");
        for (int j = 0; j < m.cols; j++) {
            printf("%Lf ", (long double) MAT_AT(m, i, j));
        }
        printf("}\n");
    }
}

#endif // Implementation