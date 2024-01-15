# CMDA 4634 Final Project: the PageRank Algorithm

For my final project in my GPU programming class, I decided to implement the PageRank algorithm used by Google to rank web pages based upon the probability of visitation by a random web surfer. Included in this repository is a C implementation and a CUDA implementation.

## Compiling the Code

To compile the C code: gcc -o pagerank pagerank.c prfuncs.c

To compile the CUDA code: make cudaPagerank

## Run the Code

To run the code: ./name-of-executable matrixDimension numEdges fileName

Note that numEdges refers to the number of non-zero entries in the adjacency matrix.

