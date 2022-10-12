#include "Math_CUDA.h"

template <class DataType>
__global__ void vector_sum_kernel(llint_t N, DataType* in1, DataType* in2)
{
	long long int i = blockIdx.x*blockDim.x+threadIdx.x;
	if(i<N)
	{
		in1[i] += in2[i];
	}
}

template <class DataType>
void vector_sum(llint_t N, DataType* in1, DataType* in2)
{
	int threads = 512;
	int blocks = N/512+1;
	vector_sum_kernel<<<blocks,threads>>>(N,in1,in2);
}

template <class DataType>
__global__ void matrix_sum_kernel(llint_t Nr, llint_t Nc, DataType* in1, DataType* in2)
{
	long long int i = blockIdx.x*blockDim.x+threadIdx.x;
	long long int j = blockIdx.y*blockDim.y+threadIdx.y;
	if(i<Nr && j<Nc)
	{
		in1[i*Nc+j] += in2[i*Nc+j];
	}
}

template <class DataType>
void martrix_sum(llint_t Nr, llint_t Nc, DataType* in1, DataType* in2)
{
	dim3 threads(512, 512);
    dim3 blocks(Nr/512+1, Nc/512+1);
	matrix_sum_kernel<<<blocks,threads>>>(Nr,Nc,in1,in2);
}

template <class DataType>
__global__ void gradient_kernel(llint_t N, DataType* in, double* out, double h)
{
	double H1 = 1/h;
	double H2 = 1/2/h;
	
	if(i==0)
	{
		out[i] = (in[i+1]-in[i])*H1;
	}
	else if(i==(N-1))
	{
		out[i] = (in[i]-in[i-1])*H1;
	}
	else
	{
		in[i] = (in[i+1]-in[i-1])*H2;
	}
}

template <class DataType>
void gradient(llint_t N, DataType* in, double* out, double h)
{
	int threads = 512;
	int blocks = N/512+1;
	gradient_kernel<<<blocks,threads>>>(N,in,out,h);
}

template <class DataType>
__global__ void gradient_general_kernel(llint_t N, DataType* in, double* out, DataType* x)
{
	if(i==0)
	{
		out[i] = (in[i+1]-in[i])/(x[i+1]-x[i]);
	}
	else if(i==(N-1))
	{
		out[i] = (in[i]-in[i-1])/(x[i]-x[i-1]);
	}
	else
	{
		in[i] = (in[i+1]-in[i-1])/(x[i+1]-x[i-1]);
	}
}

template <class DataType>
void gradient_general(llint_t N, DataType* in, double* out, DataType* x)
{
	int threads = 512;
	int blocks = N/512+1;
	gradient_kernel<<<blocks,threads>>>(N,in,out,x);
}