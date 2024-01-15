#include <time.h>
#include <math.h>
#include "prfuncs.h"

// create number of iterations counter
#define MAX_ITERS 100

// create tolerance
#define tol 0.0001

// function to perform m/n*e*e^T*x:
// sum up x1 then "scatter" into an array and multiply by m/n
// STORE result in x2
void teleportationOp (double* x1, double* x2, float m, int n) {
    // compute e^T*x1
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += x1[i];
    }

    // compute m/n*e*sum
    for (int i = 0; i < n; i++) {
        x2[i] = (m/n)*sum;
    }
}

//function to perform (1-m)ed^Tx
void danglingOp(double* x1, double* x2, int* d, int matDim, float m) {
    double sum = 0.0;
    for (int i = 0; i < matDim; i++) {
        if (d[i]) {
            sum += x1[i];
        }
    }

    for (int i = 0; i < matDim; i++) {
        x2[i] = x2[i] + ((1.0-m)/matDim)*sum;
    }
}

// function to perform sparse matrix vector multiplication of A and x
// ADD result to x2
void sparseMatVecMult (sparseMatrix* A, double* x1, double* x2, int* colSums, int n, float m) {

    for (int i = 0; i < n; i++) {
        int start =  A->rows[i];
        int end = A->rows[i+1];
        if (start != end) {
            for (int j = start; j < end; j++) {
                int col = A->cols[j];
                float val = (1.0 - m) * (1.0 / colSums[col]);
                x2[i] = x2[i] + val*x1[col];
            }
        }
    }
}

// take the l1norm of x1 and x2 and return it as the tolerance
double tolerance (double* x1, double* x2, int n) {
    double l1Diff = 0;
    for (int i = 0; i < n; i++) {
        l1Diff += fabs(x1[i] - x2[i]);
    }

    return l1Diff;

}

double pagerank (sparseMatrix* A, double* x1, double* x2, int* d, int* colSums,   
                float m, int matDim, int hasDangling) {
    teleportationOp(x1, x2, m, matDim);
    if (hasDangling) {
        danglingOp (x1, x2, d, matDim, m);
    }
    sparseMatVecMult(A, x1, x2, colSums, matDim, m);
    double t = tolerance(x1, x2, matDim);
    return t;
}

int main (int argc, char** argv) {

// check proper usage of program
if (argc < 4) {
    printf("Command usage: %s %s %s %s\n", argv[0], "n", "num_edges", "matrix.txt");
    exit(1);
}

//set dimensions of matrix
int matDim = atoi(argv[1]);
int num_edges = atoi(argv[2]);

//set m
float m = 0.15;

// calloc the d vector to be of size n
int* d = (int*)calloc(matDim,sizeof(int));

// malloc the xold vector to be of size n
double* xold = (double*)calloc(matDim,sizeof(double));

// initialize xold to be 1/n
for (int i = 0; i < matDim; i++) {
    xold[i] = 1.0/matDim;
}

// calloc xnew vector
double* xnew = (double*)calloc(matDim,sizeof(double));

// Initialize arrays for the CSR format
sparseMatrix sm;
sm.rows = (int*)calloc((matDim + 1), sizeof(int));
sm.cols = (int*)calloc(num_edges, sizeof(int));

// Initialize arrays to calculate the sum of entries in each column
int *colSums = (int*)calloc(matDim, sizeof(int));

FILE *file = fopen(argv[3], "r");
if (file == NULL) {
    printf("Error opening file");
    exit(1);
}

//read from file and initialize sparse matrix
readCSR(file, matDim, num_edges, m, &sm, colSums);

// determine how many dangling nodes the graph has
int hasDangling = isDangling(d, colSums, matDim);


// create boolean to indicate when to swap xold and xnew within the functions
int xoldFirst = 1;

// have while loop checking convergence
int numIters = 0;
clock_t tic = clock();
while (numIters < MAX_ITERS) {
    // increment number of iterations
    numIters++;
    // check boolean: if false, start with xold, else, start with xnew
    double t;
    if (xoldFirst) {
        t = pagerank(&sm, xold, xnew, d, colSums, m, matDim, hasDangling);
        xoldFirst = 0;

    }
    else {
        t = pagerank(&sm, xnew, xold, d, colSums, m, matDim, hasDangling);
        xoldFirst = 1;
    }

    if (t < tol) {
        break;
    }

}

clock_t toc = clock();
double elapsed = (double)(toc - tic) / CLOCKS_PER_SEC;

if (numIters == MAX_ITERS) {
    printf("Maximum number of iterations exceeded in %g seconds, convergence failed!\n", elapsed);
    exit(1);
}

// used to print rankings of pages 
sortingVec* x = (sortingVec*)calloc(matDim, sizeof(sortingVec));
printf("The PageRank Algorithm successfully converged in %d steps in %g seconds! Here are the resulting page rankings and probabilities:\n",numIters, elapsed);
if (xoldFirst) {
    convertToStochastic(xold, matDim);
    rankNPrint(x, xold, matDim);
}
else {
    convertToStochastic(xnew, matDim);
    rankNPrint(x, xnew, matDim);
}

//free dynamically allocated memory
free(d);
free(xold);
free(xnew);
free(sm.cols);
free(sm.rows);
free(colSums);
free(x);

return 0;
}