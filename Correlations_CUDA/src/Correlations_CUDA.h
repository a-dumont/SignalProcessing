#include <cuda.h>
#include <cuComplex.h>

void autocorrelation_cuda(int N, cuDoubleComplex* in, double* out, int blocks, int threads);
void xcorrelation_cuda(int N, cuDoubleComplex* in, cuDoubleComplex* in2, int blocks, int threads);
void reduction_complex_cuda(int N, int howmany, cuDoubleComplex* in, cuDoubleComplex* out, int size, int blocks, int threads);
//#include "Correlations_CUDA.tpp"
