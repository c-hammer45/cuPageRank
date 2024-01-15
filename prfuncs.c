#include "prfuncs.h"

//checks if a graph has a dangling node, used to create d vector
//return true or false if there is at least one dangling node
int isDangling(int* d, int* colSums, int matDim) {
    int hasDangling = 0;
    for (int i = 0; i < matDim; i++) {
    d[i] = !colSums[i];
    if (d[i]) {
        hasDangling = 1;
    }
}
return hasDangling;
}

//read from file straight into CSR format
void readCSR(FILE *file, int matDim, int num_edges, float m, sparseMatrix* sm, int* colSums) {
        // Read and process the pairs of row and column indices
    for (int i = 0; i < num_edges; i++) {
        int row, col;
        if (fscanf(file, "%d %d", &row, &col) != 2) {
            fprintf(stderr, "Error reading edge %d\n", i);
            exit(1);
        }
        // Convert 1-based indexing to 0-based indexing
        row--;
        col--;

        // Increment the row_ptr for the corresponding row
        sm->rows[row + 1]++;

        // Update the sum for the column
        colSums[col]++;
        sm->cols[i] = col;
    }
    fclose(file);

    // Calculate row_ptr values and compute the values based on the sum of column entries
    for (int i = 1; i <= matDim; i++) {
        sm->rows[i] += sm->rows[i - 1];
    }

}

// convert final result to probabilities
void convertToStochastic (double* x, int n) {
    // normalize the vector
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += x[i];
    }

    for (int i = 0; i < n; i++) {
        x[i] = x[i] / sum;
    }
}

//compare function used in qsort to sort the pages and get rankings
int comp(const void* a, const void* b) {
    sortingVec* ax = (sortingVec*)a;
    sortingVec* bx = (sortingVec*)b;

    if (ax->data > bx->data) {
        return -1;
    }
    else if (ax->data < bx->data) {
        return 1;
    }
    else {
        return 0;
    }
}

// sorts the probabilities and prints the page rankings and associated probabilities
void rankNPrint (sortingVec* x, double* vec, int matDim) {
    for (int i = 0; i < matDim; i++) {
        x[i].data = vec[i];
        x[i].index = i+1;
    }
    qsort(x, matDim, sizeof(sortingVec), comp);
    for (int i = 0; i < matDim; i++) {
        printf("(%d) %g\n", x[i].index, x[i].data);
    }
}