#include <stdlib.h>
#include <stdio.h>

// define the sparse matrix type
typedef struct sparse_type {
    int* rows;
    int* cols;
}sparseMatrix;

//define type used for sorting page rankings
typedef struct sort_type {
    int index;
    double data;
}sortingVec;

//sparse matrix function: takes in matrix and returns 2 arrays representing the sparse matrix format:
// one to hold row and column pointers
void readCSR(FILE *file, int matDim, int num_edges, float m, sparseMatrix* sm, int* colSums);

//checks if a graph has a dangling node, used to create d vector
//return true or false if there is at least one dangling node
int isDangling(int* d, int* colSums, int matDim);

// convert final result to probabilities
void convertToStochastic (double* x, int n);

//compare function used in qsort to sort the pages and get rankings
int comp(const void* a, const void* b);

// sorts the probabilities and prints the page rankings and associated probabilities
void rankNPrint (sortingVec* x, double* vec, int matDim);