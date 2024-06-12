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
    int stride;
} Mat;

#define MAT_TO(arr) ((Mat) {.mat = arr, .rows = 0, .cols = 0, .stride = 0})
#define TO_1D_MAT(arr, s) ((Mat) {.mat = arr, .rows = 1, .cols = s, .stride = s})
#define TO_2D_MAT(arr, r, c, s) ((Mat) {.mat = arr, .rows = r, .cols = c, .stride = s})
#define MAT_AT(m, i, j) (m).mat[(i) * (m).stride + (j)]
#define MAT_ROW(m, i) TO_1D_MAT(&MAT_AT(m, i, 0), (m.cols))
#define MAT_SUB(m, r, c, s) TO_2D_MAT((m).mat, r, c, s)
#define MAT_SUB_AT(m, r, c, s, sr, sc) MAT_SUB(MAT_TO(&(MAT_AT(m, sr, sc))), r, c, s)
#define MAT_PRINT(m) mat_print(#m, m)
#define MAT_SIG(m) mat_apply(m, sigmoidP);

JMATRIX_PRECISION rand_JMATRIX_PRECISION();
JMATRIX_PRECISION sigmoidP(JMATRIX_PRECISION x);

Mat mat_alloc(int rows, int cols, int stride);
void mat_init(Mat *m, int rows, int cols, int stride);
void mat_fill(Mat m, JMATRIX_PRECISION val);
void mat_copy(Mat dst, Mat src);
void mat_rand(Mat m, JMATRIX_PRECISION low, JMATRIX_PRECISION high);
void mat_dot(Mat out, Mat a, Mat b);
void mat_sum(Mat out, Mat a);
void mat_apply(Mat m, JMATRIX_PRECISION (*f)(JMATRIX_PRECISION));
void mat_print(const char *name, Mat a);

#endif // Header

#ifdef JMATRIX_IMPLEMENTATION
#undef JMATRIX_IMPLEMENTATION

/**
 * @return Random number between 0 and 1
 */
JMATRIX_PRECISION rand_JMATRIX_PRECISION() {
    return (JMATRIX_PRECISION) rand() / (JMATRIX_PRECISION) RAND_MAX;
}

/**
 * @brief Passes a number through the sigmoid function
 * @param x is the input number
 * @return Number passed through the function
 */
JMATRIX_PRECISION sigmoidP(JMATRIX_PRECISION x) {
    return 1. / (1. + exp(x));
}

/**
 * @brief Allocates the array of a matrix with a specified size
 * @param rows is the amount of rows
 * @param cols is the amount of columns
 * @param stride is the stride
 * @return Matrix of specified size
 */
Mat mat_alloc(int rows, int cols, int stride) {
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.stride = stride;
    m.mat = JMATRIX_MALLOC(sizeof(JMATRIX_PRECISION) * rows * cols);
    JMATRIX_ASSERT(m.mat != NULL);
    return m;
}

/**
 * @brief Initializes a pre-allocated matrix
 * @param m is the pointer to the matrix
 * @param rows is the amount of rows
 * @param cols is the amount of columns
 * @param stride is the stride
 */
void mat_init(Mat *m, int rows, int cols, int stride) {
    m->rows = rows;
    m->cols = cols;
    m->stride = stride;
    m->mat = JMATRIX_MALLOC(sizeof(JMATRIX_PRECISION) * rows * cols);
    JMATRIX_ASSERT(m->mat != NULL);
}

/**
 * @brief Fills a matrix with a number
 * @param m is the matrix
 * @param val is the number
 */
void mat_fill(Mat m, JMATRIX_PRECISION val) {
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            MAT_AT(m, i, j) = val;
        }
    }
}

/**
 * @brief Copies all values from a matrix into another matrix
 * @param dst is the destination matrix
 * @param src is the source matrix
 */
void mat_copy(Mat dst, Mat src) {
    JMATRIX_ASSERT(dst.rows == src.rows);
    JMATRIX_ASSERT(dst.cols == src.cols);
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            MAT_AT(dst, i, j) = MAT_AT(src, i, j);
        }
    }
}

/**
 * @brief Fills a matrix with a random number from low to high
 * @param low is the minimum number
 * @param high is the maximum number
 */
void mat_rand(Mat m, JMATRIX_PRECISION low, JMATRIX_PRECISION high) {
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            MAT_AT(m, i, j) = rand_JMATRIX_PRECISION() * (high - low) + low;
        }
    }
}

/**
 * @brief Applies the dot product to two matricies and stores the result in a third matrix
 * @param out is the destination matrix
 * @param a is the first matrix
 * @param b is the second matrix
 */
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

/**
 * @brief Adds two matricies and stores the result in the first matrix
 * @param out is the destination matrix
 * @param m is the matrix that gets added
 */
void mat_sum(Mat out, Mat m) {
    JMATRIX_ASSERT(out.rows == m.rows);
    JMATRIX_ASSERT(out.cols == m.cols);
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            MAT_AT(out, i, j) += MAT_AT(m, i, j);
        }
    }
}

/**
 * @brief Applies a function to every element in a matrix
 * @param m is the matrix
 * @param f is the function pointer
 * @warning f needs to take a number as input and nothing else
 */
void mat_apply(Mat m, JMATRIX_PRECISION (*f)(JMATRIX_PRECISION)) {
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            MAT_AT(m, i, j) = (*f)(MAT_AT(m, i, j));
        }
    }
}

/**
 * @brief Prints the contents of a matrix after printing its name
 * @param name is the string storing the name
 * @param m is the matrix
 */
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