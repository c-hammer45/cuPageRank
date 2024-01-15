# CMDA 4634 Final Project: the PageRank Algorithm

For my final project, I decided to implement the PageRank algorithm used by Google to rank web pages based upon the probability of visitation by a random web surfer. Included in this repository is a C implementation, a CUDA implementation, a makefile, and some benchmark datasets.

## Compiling the Code

To compile the C code: gcc -o pagerank pagerank.c prfuncs.c

To compile the CUDA code: make cudaPagerank

## Run the Code

To run the code: ./name-of-executable matrixDimension numEdges fileName

