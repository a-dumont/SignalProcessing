#include "Math_CUDA.h"

template <class DataType>
__global__ void vector_sum_kernel(long long int N, DataType* in1, DataType* in2)
{
	long long int i = blockIdx.x*blockDim.x+threadIdx.x;
	if(i<N)
	{
		in1[i] += in2[i];
	}
}

template <class DataType>
void vector_sum(long long int N, DataType* in1, DataType* in2)
{
	int threads = 512;
	int blocks = N/512+1;
	vector_sum_kernel<<<blocks,threads>>>(N,in1,in2);
}
template void vector_sum<float>(long long int N, float* in1, float* in2);
template void vector_sum<double>(long long int N, double* in1, double* in2);
template void vector_sum<int8_t>(long long int N, int8_t* in1, int8_t* in2);
template void vector_sum<int16_t>(long long int N, int16_t* in1, int16_t* in2);
template void vector_sum<int32_t>(long long int N, int32_t* in1, int32_t* in2);
template void vector_sum<int64_t>(long long int N, int64_t* in1, int64_t* in2);
template void vector_sum<uint8_t>(long long int N, uint8_t* in1, uint8_t* in2);
template void vector_sum<uint16_t>(long long int N, uint16_t* in1, uint16_t* in2);
template void vector_sum<uint32_t>(long long int N, uint32_t* in1, uint32_t* in2);
template void vector_sum<uint64_t>(long long int N, uint64_t* in1, uint64_t* in2);

template <class DataType>
__global__ void vector_product_kernel(long long int N, DataType* in1, DataType* in2)
{
	long long int i = blockIdx.x*blockDim.x+threadIdx.x;
	if(i<N)
	{
		in1[i] *= in2[i];
	}
}

template <class DataType>
void vector_product(long long int N, DataType* in1, DataType* in2)
{
	int threads = 512;
	int blocks = N/512+1;
	vector_product_kernel<<<blocks,threads>>>(N,in1,in2);
}
template void vector_product<float>(long long int N, float* in1, float* in2);
template void vector_product<double>(long long int N, double* in1, double* in2);
template void vector_product<int8_t>(long long int N, int8_t* in1, int8_t* in2);
template void vector_product<int16_t>(long long int N, int16_t* in1, int16_t* in2);
template void vector_product<int32_t>(long long int N, int32_t* in1, int32_t* in2);
template void vector_product<int64_t>(long long int N, int64_t* in1, int64_t* in2);
template void vector_product<uint8_t>(long long int N, uint8_t* in1, uint8_t* in2);
template void vector_product<uint16_t>(long long int N, uint16_t* in1, uint16_t* in2);
template void vector_product<uint32_t>(long long int N, uint32_t* in1, uint32_t* in2);
template void vector_product<uint64_t>(long long int N, uint64_t* in1, uint64_t* in2);

template <class DataType>
__global__ void vector_diff_kernel(long long int N, DataType* in1, DataType* in2)
{
	long long int i = blockIdx.x*blockDim.x+threadIdx.x;
	if(i<N)
	{
		in1[i] -= in2[i];
	}
}

template <class DataType>
void vector_diff(long long int N, DataType* in1, DataType* in2)
{
	int threads = 512;
	int blocks = N/512+1;
	vector_diff_kernel<<<blocks,threads>>>(N,in1,in2);
}
template void vector_diff<float>(long long int N, float* in1, float* in2);
template void vector_diff<double>(long long int N, double* in1, double* in2);
template void vector_diff<int8_t>(long long int N, int8_t* in1, int8_t* in2);
template void vector_diff<int16_t>(long long int N, int16_t* in1, int16_t* in2);
template void vector_diff<int32_t>(long long int N, int32_t* in1, int32_t* in2);
template void vector_diff<int64_t>(long long int N, int64_t* in1, int64_t* in2);
template void vector_diff<uint8_t>(long long int N, uint8_t* in1, uint8_t* in2);
template void vector_diff<uint16_t>(long long int N, uint16_t* in1, uint16_t* in2);
template void vector_diff<uint32_t>(long long int N, uint32_t* in1, uint32_t* in2);
template void vector_diff<uint64_t>(long long int N, uint64_t* in1, uint64_t* in2);

template <class DataType>
__global__ void vector_div_kernel(long long int N, DataType* in1, DataType* in2)
{
	long long int i = blockIdx.x*blockDim.x+threadIdx.x;
	if(i<N)
	{
		in1[i] /= in2[i];
	}
}

template <class DataType>
void vector_div(long long int N, DataType* in1, DataType* in2)
{
	int threads = 512;
	int blocks = N/512+1;
	vector_div_kernel<<<blocks,threads>>>(N,in1,in2);
}
template void vector_div<float>(long long int N, float* in1, float* in2);
template void vector_div<double>(long long int N, double* in1, double* in2);
template void vector_div<int8_t>(long long int N, int8_t* in1, int8_t* in2);
template void vector_div<int16_t>(long long int N, int16_t* in1, int16_t* in2);
template void vector_div<int32_t>(long long int N, int32_t* in1, int32_t* in2);
template void vector_div<int64_t>(long long int N, int64_t* in1, int64_t* in2);
template void vector_div<uint8_t>(long long int N, uint8_t* in1, uint8_t* in2);
template void vector_div<uint16_t>(long long int N, uint16_t* in1, uint16_t* in2);
template void vector_div<uint32_t>(long long int N, uint32_t* in1, uint32_t* in2);
template void vector_div<uint64_t>(long long int N, uint64_t* in1, uint64_t* in2);

template <class DataType>
__global__ void matrix_sum_kernel(long long int Nr, long long int Nc, DataType* in1, DataType* in2)
{
	long long int i = blockIdx.x*blockDim.x+threadIdx.x;
	long long int j = blockIdx.y*blockDim.y+threadIdx.y;
	if(i<Nr && j<Nc)
	{
		in1[i*Nc+j] += in2[i*Nc+j];
	}
}

template <class DataType>
void matrix_sum(long long int Nr, long long int Nc, DataType* in1, DataType* in2)
{
	dim3 threads(512, 512);
    dim3 blocks(Nr/512+1, Nc/512+1);
	matrix_sum_kernel<<<blocks,threads>>>(Nr,Nc,in1,in2);
}

template <class DataType>
__global__ void matrix_prod_kernel(long long int Nr, long long int Nc, DataType* in1, DataType* in2)
{
	long long int i = blockIdx.x*blockDim.x+threadIdx.x;
	long long int j = blockIdx.y*blockDim.y+threadIdx.y;
	if(i<Nr && j<Nc)
	{
		in1[i*Nc+j] *= in2[i*Nc+j];
	}
}

template <class DataType>
void matrix_prod(long long int Nr, long long int Nc, DataType* in1, DataType* in2)
{
	dim3 threads(512, 512);
    dim3 blocks(Nr/512+1, Nc/512+1);
	matrix_prod_kernel<<<blocks,threads>>>(Nr,Nc,in1,in2);
}

template <class DataType>
__global__ void matrix_diff_kernel(long long int Nr, long long int Nc, DataType* in1, DataType* in2)
{
	long long int i = blockIdx.x*blockDim.x+threadIdx.x;
	long long int j = blockIdx.y*blockDim.y+threadIdx.y;
	if(i<Nr && j<Nc)
	{
		in1[i*Nc+j] -= in2[i*Nc+j];
	}
}

template <class DataType>
void matrix_diff(long long int Nr, long long int Nc, DataType* in1, DataType* in2)
{
	dim3 threads(512, 512);
    dim3 blocks(Nr/512+1, Nc/512+1);
	matrix_diff_kernel<<<blocks,threads>>>(Nr,Nc,in1,in2);
}

template <class DataType>
__global__ void matrix_div_kernel(long long int Nr, long long int Nc, DataType* in1, DataType* in2)
{
	long long int i = blockIdx.x*blockDim.x+threadIdx.x;
	long long int j = blockIdx.y*blockDim.y+threadIdx.y;
	if(i<Nr && j<Nc)
	{
		in1[i*Nc+j] /= in2[i*Nc+j];
	}
}

template <class DataType>
void matrix_div(long long int Nr, long long int Nc, DataType* in1, DataType* in2)
{
	dim3 threads(512, 512);
    dim3 blocks(Nr/512+1, Nc/512+1);
	matrix_div_kernel<<<blocks,threads>>>(Nr,Nc,in1,in2);
}

template <class DataType>
__global__ void gradient_kernel(long long int N, DataType* in, double* out, double h)
{
	double H1 = 1/h;
	double H2 = 1/2/h;

	long long int i = blockIdx.x*blockDim.x+threadIdx.x;	
	
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
void gradient(long long int N, DataType* in, double* out, double h)
{
	int threads = 512;
	int blocks = N/512+1;
	gradient_kernel<<<blocks,threads>>>(N,in,out,h);
}

template <class DataType>
__global__ void gradient_general_kernel(long long int N, DataType* in, double* out, DataType* x)
{
	long long int i = blockIdx.x*blockDim.x+threadIdx.x;	
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
void gradient_general(long long int N, DataType* in, double* out, DataType* x)
{
	int threads = 512;
	int blocks = N/512+1;
	gradient_kernel<<<blocks,threads>>>(N,in,out,x);
}
