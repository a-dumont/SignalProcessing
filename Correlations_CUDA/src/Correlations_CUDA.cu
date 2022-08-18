#include "Correlations_CUDA.h"

__global__ void autocorrelation_cuda_kernel(int N, cuDoubleComplex* in, double* out, int threads)
{
	// Compute the correlation
	int i = threadIdx.x+blockIdx.x*threads;
	if(i<N)
	{
		out[i] = cuCreal(cuCmul(in[i],cuConj(in[i])));
	}
}

void autocorrelation_cuda(int N, cuDoubleComplex* in, double* out, int blocks, int threads)
{
	autocorrelation_cuda_kernel<<<blocks,threads>>>(N,in,out,threads);
	cudaDeviceSynchronize();
}

__global__ void xcorrelation_cuda_kernel(int N, cuDoubleComplex* in, cuDoubleComplex* in2, int threads)
{
	// Compute the correlation
	int i = threadIdx.x+blockIdx.x*threads;
	if(i<N)
	{
		in[i] = cuCmul(in[i],cuConj(in2[i]));
	}
}

void xcorrelation_cuda(int N, cuDoubleComplex* in, cuDoubleComplex* in2, int blocks, int threads)
{
	xcorrelation_cuda_kernel<<<blocks,threads>>>(N,in,in2,threads);
	cudaDeviceSynchronize();
}

__global__ void reduction_complex_kernel(cuDoubleComplex* in, cuDoubleComplex* out, int jump, int threads)
{
		int i = threads*blockIdx.x+threadIdx.x;
		if(i<jump)
		{
			out[i] = cuCadd(in[i],in[i+jump]);
		}
}

__global__ void product_complex_kernel(cuDoubleComplex* in, double N, int size,int threads)
{
		int i = threads*blockIdx.x+threadIdx.x;
		if(i<size)
		{
			in[i] = make_cuDoubleComplex(cuCreal(in[i])/N,cuCimag(in[i])/N);
		}

}

void reduction_complex_cuda(int N, int howmany, cuDoubleComplex* in, cuDoubleComplex* out, int size, int blocks, int threads)
{
	int jump = N/2*(size/2+1);
	reduction_complex_kernel<<<blocks,threads>>>(in,out,jump,threads);
	if(N/2 >= 1)
	{
		reduction_complex_cuda(N/2,howmany,in,out,size,(blocks-1)/2+1,threads);
	}
	else
	{
		product_complex_kernel<<<size/512+1,512>>>(in,howmany,size,threads);
		cudaDeviceSynchronize();
	}
}
