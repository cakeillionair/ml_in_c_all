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
#define NN_PRINT(n) nn_print(#n, n, 4)

NN nn_alloc(int size, int inSize, int *lSizes);
Mat *nn_datafile_alloc(FILE *f);
void nn_init(NN *n, Mat *mem, Mat *paramMem, Mat *outMem, int size, int inSize, int *lSizes);
void nn_fill(NN n, JMATRIX_PRECISION val);
void nn_copy(NN dst, NN src, bool out);
void nn_rand(NN n, JMATRIX_PRECISION low, JMATRIX_PRECISION high);
void nn_forward(NN n, Mat in, JMATRIX_PRECISION (*f)(JMATRIX_PRECISION));
JMATRIX_PRECISION nn_get_cost(NN n, Mat in, Mat out, JMATRIX_PRECISION (*f)(JMATRIX_PRECISION));
void nn_finite_diff(NN g, NN n, Mat in, Mat out, JMATRIX_PRECISION (*f)(JMATRIX_PRECISION), JMATRIX_PRECISION eps);
void nn_learn(NN out, NN g, JMATRIX_PRECISION rate);
void nn_print(const char *name, NN n, int indent);

#endif // Header

#ifdef JNETWORK_IMPLEMENTATION
#undef JNETWORK_IMPLEMENTATION

/**
 * @brief Allocates a neural network of the specified size
 * @param size is the amount of layers
 * @param inSize is the size of the inputs the network can take
 * @param lSizes is the array of layer sizes
 * @return Neural network of specified size
 */
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
    JMATRIX_ASSERT(n.w != NULL && n.b != NULL && n.out != NULL);
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

#ifndef MAX_DATA_SIZE
#define MAX_DATA_SIZE 640
#endif

/**
 * @brief Loads nn training data from a file into a matrix
 * @param f is the file pointer
 * @return An array of matricies of size two containing the data split up into input and output matricies
 */
Mat *nn_datafile_alloc(FILE *f) {
    int rows, inCols, outCols;
    JMATRIX_ASSERT(fscanf(f, "%d;%d;%d;", &rows, &inCols, &outCols) != EOF);
    int size = rows * (inCols + outCols);
    JMATRIX_ASSERT(size <= MAX_DATA_SIZE);

    Mat *result = JMATRIX_MALLOC(sizeof(Mat) * 2);
    JMATRIX_PRECISION *mat = JMATRIX_MALLOC(sizeof(JMATRIX_PRECISION) * size);
    JMATRIX_ASSERT(result != NULL && mat != NULL);
    result[0].mat = mat;
    result[0].rows = rows;
    result[0].cols = inCols;
    result[0].stride = inCols + outCols;
    result[1].mat = &mat[inCols];
    result[1].rows = rows;
    result[1].cols = outCols;
    result[1].stride = inCols + outCols;

    for (int i = 0; i < size; i++) {
        long double buffer;
        if (fscanf(f, "%Lf,", &buffer) == EOF) {
            free(result);
            free(mat);
            fclose(f);
            return NULL;
        }
        mat[i] = buffer;
    }

    return result;
}

/**
 * @brief Initializes a pre-allocated neural network
 * @param n is the pointer to the nn
 * @param mem is an array of matricies that the nn is going to store its matricies in
 * @param paramMem is a pointer to a matrix containing pre-allocated space for the weights and biases
 * @param outMem is a pointer to a matrix containing pre-allocated space for the outputs of each layer
 * @param size is the amount of layers of the nn
 * @param inSize is the size of the inputs the network can take
 * @param lSizes is the array of layer sizes
 */
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

/**
 * @brief Fills a neural network with a number
 * @param n is the nn
 * @param val is the number
 */
void nn_fill(NN n, JMATRIX_PRECISION val) {
    for (int i = 0; i < n.size; i++) {
        mat_fill(n.w[i], val);
        mat_fill(n.b[i], val);
        mat_fill(n.out[i], val);
    }
}

/**
 * @brief Copies a nn into another
 * @param dst is the destination nn
 * @param src is the source nn
 * @param out is a bool that controls whether the output layers are copied
 */
void nn_copy(NN dst, NN src, bool out) {
    JMATRIX_ASSERT(dst.size = src.size);
    for (int i = 0; i < src.size; i++) {
        mat_copy(dst.w[i], src.w[i]);
        mat_copy(dst.b[i], src.b[i]);
        if (out) mat_copy(dst.out[i], src.out[i]);
    }
}

/**
 * @brief Randomizes all values in a neural network
 * @param n is the nn
 * @param low is the minimum value
 * @param high is the maximum value
 */
void nn_rand(NN n, JMATRIX_PRECISION low, JMATRIX_PRECISION high) {
    for (int i = 0; i < n.size; i++) {
        mat_rand(n.w[i], low, high);
        mat_rand(n.b[i], low, high);
        mat_rand(n.out[i], low, high);
    }
}

/**
 * @brief Forwards inputs from a matrix through a neural network
 * @param n is the nn
 * @param in is the input matrix
 * @param f is the function that gets applied after each layer
 */
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

/**
 * @brief Calculates a cost of a neural network
 * @param n is the nn
 * @param in is the input matrix
 * @param out is the matrix with correct output values
 * @param f is the function the nn is forwarded with
 * @deprecated This function is not good and a better one is on the way
 */
JMATRIX_PRECISION nn_get_cost(NN n, Mat in, Mat out, JMATRIX_PRECISION (*f)(JMATRIX_PRECISION)) {
    JMATRIX_ASSERT(in.cols == n.w[0].rows);
    JMATRIX_ASSERT(in.rows == out.rows);
    JMATRIX_ASSERT(out.cols == NN_GET_OUT(n).cols);

    JMATRIX_PRECISION result = 0;

    for (int i = 0; i < in.rows; i++) {
        nn_forward(n, MAT_ROW(in, i), f);
        JMATRIX_PRECISION dTotal = 0;
        for (int j = 0; j < out.cols; j++) {
            JMATRIX_PRECISION diff = MAT_AT(NN_GET_OUT(n), /*F*CK THIS NUMBER -->*/0, j) - MAT_AT(out, i, j);
            dTotal += diff * diff;/*                         It broke my code and it took me HOURS*/
        }/*                                                  to find the error!!!                 */
        result += dTotal / out.cols;
    }

    return result / in.rows;
}

#define NN_COST (nn_get_cost(n, in, out, f))

/**
 * @brief Calculates a gradient by which a neural network needs to be changed
 * @param g is the gradient nn
 * @param n is the input nn
 * @param in is the input matrix
 * @param out is the matrix with correct output values
 * @param f is the function the nn is forwarded with
 * @param eps is the value by which the parameters are changed by
 * @deprecated This function is not good and a better one is on the way
 */
void nn_finite_diff(NN g, NN n, Mat in, Mat out, JMATRIX_PRECISION (*f)(JMATRIX_PRECISION), JMATRIX_PRECISION eps) {
    JMATRIX_ASSERT(g.size == n.size);
    JMATRIX_PRECISION buffer;

    JMATRIX_PRECISION cost = NN_COST;

    for (int i = 0; i < n.size; i++) {
        for (int j = 0; j < n.w[i].cols; j++) {
            for (int k = 0; k < n.w[i].rows; k++) {
                buffer = MAT_AT(n.w[i], k, j);
                MAT_AT(n.w[i], k, j) += eps;
                MAT_AT(g.w[i], k, j) = (NN_COST - cost) / eps;
                MAT_AT(n.w[i], k, j) = buffer;
            }
            buffer = MAT_AT(n.b[i], 0, j);
            MAT_AT(n.b[i], 0, j) += eps;
            MAT_AT(g.b[i], 0, j) = (NN_COST - cost) / eps;
            MAT_AT(n.b[i], 0, j) = buffer;
        }
    }
}

/**
 * @brief Applies a gradient to a neural network
 * @param out is the nn
 * @param g is the gradient nn
 * @param rate is the learning rate
 */
void nn_learn(NN out, NN g, JMATRIX_PRECISION rate) {
    JMATRIX_ASSERT(g.size == out.size);
    JMATRIX_ASSERT(rate != 0);

    for (int i = 0; i < out.size; i++) {
        for (int j = 0; j < out.w[i].cols; j++) {
            for (int k = 0; k < out.w[i].rows; k++) {
                MAT_AT(out.w[i], k, j) -= rate * MAT_AT(g.w[i], k, j);
            }
            MAT_AT(out.b[i], 0, j) -= rate * MAT_AT(g.b[i], 0, j);
        }
    }
}

/**
 * @brief Prints the matricies of a neural network with its name
 * @param name is the name
 * @param n is the nn
 * @param indent is the amount of indentation
 */
void nn_print(const char *name, NN n, int indent) {
    JMATRIX_ASSERT(indent <= JMATRIX_INDENT_LIMIT);
    printf("Network: %s\n", name);
    for (int i = 0; i < n.size; i++) {
        printf("%*sLayer %i:\n", indent, "", i);
        printf("%*s", 2 * indent, "");
        mat_print("Weights", n.w[i], 3 * indent);
        printf("%*s", 2 * indent, "");
        mat_print("Biases", n.b[i], 3 * indent);
        printf("%*s", 2 * indent, "");
        mat_print("Outputs", n.out[i], 3 * indent);
    }
}

#endif // Implementation