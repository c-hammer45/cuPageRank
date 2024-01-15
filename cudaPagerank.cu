#include <time.h>
#include <math.h>
#include <cuda.h>
extern "C" {
    #include "prfuncs.h"
}


// create number of iterations counter
#define MAX_ITERS 100

// create tolerance
#define toler 0.0001

//performs sparse matrix vector multiplication (1-m)Ax using row and column pointer arrays
//no need to store the actual values when can be calculated on the fly
__global__
void sparseMatVecMult (int* cols, int* rows, double* x1, double* x2, int* colSums, float m, int n) {
    
    int threadId = threadIdx.x + blockDim.x*blockIdx.x;

    if (threadId < n) {
        int start = rows[threadId];
        int end = rows[threadId + 1];
        if (start != end) {
            for (int i = start; i < end; i++) {
                int col = cols[i];
                float val = (1.0 - m) * (1.0 / colSums[col]);
                x2[threadId] = x2[threadId] + val*x1[col];
            }
        }

    }
}

//Sum up values in x array in preparation for the teleportation operation
__global__ void reduction(double *x, double *result, int n) {

    int t = threadIdx.x;
    int b = blockIdx.x;
    int d = blockDim.x;
    int threadId = t + b * d;

    __shared__ double s_x[1024]; 

    s_x[t] = (threadId < n) ? x[threadId] : 0;
    __syncthreads();

    for (int n = d / 2; n > 0; n /= 2) {
        if (t < n) {
            s_x[t] += s_x[t + n];
        }
        __syncthreads();
    }

    if (t == 0) {
        atomicAdd(result, s_x[0]);
    }
}

//Uses result from reduction kernel to perform m/n*e*e^T*x
__global__
void teleportationOp (double* x2, float m, int n, double* sum) {
    
    int threadId = threadIdx.x + blockIdx.x*blockDim.x;

    // scatter sum*(m/n) and store in x2
    if (threadId < n) {
        double result = *sum;
        x2[threadId] = result * (m/n);
    }
}

//Similar to reduction but takes into account 1s vs 0s in dangling vector d
__global__
void danglingReduction (int* d, double* x, int n, double* result) {

    int t = threadIdx.x;
    int b = blockIdx.x;
    int bd = blockDim.x;
    int threadId = t + b * bd;

    __shared__ double s_x[1024]; 

    s_x[t] = (threadId < n && d[threadId]) ? x[threadId] : 0;
    __syncthreads();

    for (int n = bd / 2; n > 0; n /= 2) {
        if (t < n) {
            s_x[t] += s_x[t + n];
        }
        __syncthreads();
    }

    if (t == 0) {
        atomicAdd(result, s_x[0]);
    }

}

//uses result from dangling reduction kernel to perform (1-m)/ned^Tx
__global__
void danglingOp (double* x, float m, int n, double* result) {
    int threadId = threadIdx.x + blockIdx.x*blockDim.x;

    if (threadId < n) {
        double sum = *result;
        x[threadId] = x[threadId] + ((1.0f-m)/n)*sum;
    }
}

//compute the l1 norm between the previous and current PageRank vectors
__global__
void tolerance (double* x1, double* x2, int n, double* tol) {

    int t = threadIdx.x;
    int b = blockIdx.x;
    int bd = blockDim.x;
    int threadId = t + b * bd;

    __shared__ double s_x[1024]; 

    s_x[t] = (threadId < n) ? fabs(x1[threadId] - x2[threadId]) : 0;
    __syncthreads();

    for (int n = bd / 2; n > 0; n /= 2) {
        if (t < n) {
            s_x[t] += s_x[t + n];
        }
        __syncthreads();
    }

    if (t == 0) {
        atomicAdd(tol, s_x[0]);
    }
}

//launcher function for PageRank kernels that performs the entire calculation: (1-m)Ax + (1-m)ed^Tx + (m/n)ee^Tx 
double pagerank (int* c_rows, int* c_cols, double* c_xold, double* c_xnew, int* c_d, int* c_colSums,   
                float m, int n, double* c_result, double* c_tol, double* h_tol, int hasDangling) {
    
    int B = 512;
    int G = (n+B-1)/B;
    
    cudaMemset(c_result, 0, sizeof(double));
    reduction <<< G, B >>> (c_xold, c_result, n);
    teleportationOp <<< G, B >>> (c_xnew, m, n, c_result);
    if (hasDangling) {
        cudaMemset(c_result, 0, sizeof(double));
        danglingReduction <<< G, B >>> (c_d, c_xold, n, c_result);
        danglingOp <<< G, B >>> (c_xnew, m, n, c_result);
    }
    sparseMatVecMult <<< G, B >>> (c_cols, c_rows, c_xold, c_xnew, c_colSums, m, n);
    cudaMemset(c_tol, 0, sizeof(double));
    tolerance <<< G, B >>> (c_xold, c_xnew, n, c_tol);
    cudaMemcpy(h_tol, c_tol, sizeof(double),cudaMemcpyDeviceToHost);
    return *h_tol;
}

int main(int argc, char** argv) {

    cudaSetDevice(0);

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

    // Initialize arrays to calculate the sum of entries in each column to make matrix column-stochastic
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

    // host tolerance
    double* h_tol = (double*)calloc(1,sizeof(double));

    // initialize device fields from struct
    int* c_cols;
    int* c_rows;
    double* c_xold;
    double* c_xnew;
    int *c_d;
    int* c_colSums;
    double* c_result;
    double* c_tol;

    //cuda malloc those fields
    cudaMalloc(&c_cols, num_edges*sizeof(int));
    cudaMalloc(&c_rows, (matDim+1)*sizeof(int));
    cudaMalloc(&c_xold, matDim*sizeof(double));
    cudaMalloc(&c_xnew, matDim*sizeof(double));
    cudaMalloc(&c_colSums, matDim*sizeof(int));
    cudaMalloc(&c_d, matDim*sizeof(int));
    cudaMalloc(&c_result, sizeof(double));
    cudaMalloc(&c_tol, sizeof(double));

    // copy to device
    cudaMemcpy(c_cols, sm.cols, num_edges*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(c_rows, sm.rows, (matDim+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(c_xold, xold, matDim*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(c_xnew, xnew, matDim*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(c_colSums, colSums, matDim*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(c_d, d, matDim*sizeof(int), cudaMemcpyHostToDevice);

    // create boolean to indicate when to swap xold and xnew within the functions
    int xoldFirst = 1;

    int numIters = 0;
    
    cudaDeviceSynchronize();
    clock_t tic = clock();
    while (numIters < MAX_ITERS) {
    // increment number of iterations
    numIters++;
    
    double t;
    // check boolean: if false, start with xold, else, start with xnew
    if (xoldFirst) {
        t = pagerank(c_rows, c_cols, c_xold, c_xnew, c_d, c_colSums, m, matDim, c_result, c_tol, h_tol, hasDangling);
        xoldFirst = 0;

    }
    else {
        t = pagerank(c_rows, c_cols, c_xnew, c_xold, c_d, c_colSums, m, matDim, c_result, c_tol, h_tol, hasDangling);
        xoldFirst = 1;
    }

    //check tolerance and break if convergence is met
    if (t < toler) {
        break;
    }

}

cudaDeviceSynchronize();
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
        cudaMemcpy(xold, c_xold, matDim*sizeof(double), cudaMemcpyDeviceToHost);
        convertToStochastic(xold, matDim);
        rankNPrint(x, xold, matDim);
    }
    else {
        cudaMemcpy(xnew, c_xnew, matDim*sizeof(double), cudaMemcpyDeviceToHost);
        convertToStochastic(xnew, matDim);
        rankNPrint(x, xnew, matDim);
    }

    //free dynamically allocated memory on host
    free(d);
    free(xold);
    free(xnew);
    free(sm.cols);
    free(sm.rows);
    free(colSums);
    free(x);
    free(h_tol);

    //free dynamically allocated memory on device
    cudaFree(c_cols);
    cudaFree(c_rows);
    cudaFree(c_xold);
    cudaFree(c_xnew);
    cudaFree(c_d);
    cudaFree(c_colSums);
    cudaFree(c_result);
    cudaFree(c_tol);

    return 0;
}