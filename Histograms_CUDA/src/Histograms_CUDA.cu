#include "Histograms_CUDA.cuh"

__global__ void filter_kernel(long long int N, double* signal, double* filter, int threads)
{
	int i = threadIdx.x+blockIdx.x*threads;
	if(i<N)
	{ 
		signal[2*i] *= filter[i];
		signal[2*i+1] *= filter[i];
	}
}

__global__ void filter_kernelf(long long int N, float* signal, float* filter, int threads)
{
	int i = threadIdx.x+blockIdx.x*threads;
	if(i<N)
	{ 
		signal[2*i] *= filter[i];
		signal[2*i+1] *= filter[i];
	}
}

template<class DataType>
void filter_CUDA(long long int N, DataType* signal, DataType* filter, cudaStream_t stream){}

template<>
void filter_CUDA<double>(long long int N, double* signal, double* filter, cudaStream_t stream)
{
	int threads = 512;
	long long int blocks = N/threads;
	filter_kernel<<<blocks+1,threads,0,stream>>>(N,signal,filter,threads);
}

template<>
void filter_CUDA<float>(long long int N, float* signal, float* filter, cudaStream_t stream)
{
	int threads = 512;
	long long int blocks = N/threads;
	filter_kernelf<<<blocks+1,threads,0,stream>>>(N,signal,filter,threads);
}

template<class DataType>
__global__ void convert_kernel(long long int N, DataType* in, double* out,
								double conv, DataType offset, int threads)
{
	int i = threadIdx.x+blockIdx.x*threads;
	if(i<N)
	{
			out[i] = conv*in[i]-conv*offset;
		}
}

template<class DataType>
__global__ void convert_kernelf(long long int N, DataType* in, float* out,
								float conv, DataType offset, int threads)
{
	int i = threadIdx.x+blockIdx.x*threads;
	if(i<N)
	{
			out[i] = conv*in[i]-conv*offset;
		}
}

template<class DataType, class DataType2>
void convert(long long int N, DataType* in, DataType2* out,
								DataType2 conv, DataType offset, cudaStream_t stream){}

template<>
void convert<uint8_t,double>(long long int N, uint8_t* in, double* out,
								double conv, uint8_t offset, cudaStream_t stream)
{
	convert_kernel<uint8_t><<<(N/512)+1,512,0,stream>>>(N,in,out,conv,offset,512);
}

template<>
void convert<uint16_t,double>(long long int N, uint16_t* in, double* out,
								double conv, uint16_t offset, cudaStream_t stream)
{
	convert_kernel<uint16_t><<<(N/512)+1,512,0,stream>>>(N,in,out,conv,offset,512);
}

template<>
void convert<int16_t,double>(long long int N, int16_t* in, double* out,
								double conv, int16_t offset, cudaStream_t stream)
{
	convert_kernel<int16_t><<<(N/512)+1,512,0,stream>>>(N,in,out,conv,offset,512);
}

template<>
void convert<uint8_t,float>(long long int N, uint8_t* in, float* out,
								float conv, uint8_t offset, cudaStream_t stream)
{
	convert_kernelf<uint8_t><<<(N/512)+1,512,0,stream>>>(N,in,out,conv,offset,512);
}

template<>
void convert<uint16_t,float>(long long int N, uint16_t* in, float* out,
								float conv, uint16_t offset, cudaStream_t stream)
{
	convert_kernelf<uint16_t><<<(N/512)+1,512,0,stream>>>(N,in,out,conv,offset,512);
}

template<>
void convert<int16_t,float>(long long int N, int16_t* in, float* out,
								float conv, int16_t offset, cudaStream_t stream)
{
	convert_kernelf<int16_t><<<(N/512)+1,512,0,stream>>>(N,in,out,conv,offset,512);
}

template<class DataType>
__global__ void rconvert_kernel(long long int N, double* in, DataType* out, 
				double conv, DataType offset, int threads)
{
	int i = threadIdx.x+blockIdx.x*threads;
	if(i<N)
	{
			out[i] = (DataType) (in[i]*conv+1.0*offset);
	}
}

template<class DataType>
__global__ void rconvert_kernelf(long long int N, float* in, DataType* out, 
				float conv, DataType offset, int threads)
{
	int i = threadIdx.x+blockIdx.x*threads;
	if(i<N)
	{
			out[i] = (DataType) (in[i]*conv+1.0*offset);
	}
}

template<class DataType, class DataType2>
void rconvert(long long int N, DataType2* in, DataType* out,
								DataType2 conv, DataType offset, cudaStream_t stream){}

template<>
void rconvert<uint8_t,double>(long long int N, double* in, uint8_t* out,
								double conv, uint8_t offset, cudaStream_t stream)
{
	rconvert_kernel<uint8_t><<<(N/512)+1,512,0,stream>>>(N,in,out,conv,offset,512);
}

template<>
void rconvert<uint16_t,double>(long long int N, double* in, uint16_t* out,
								double conv, uint16_t offset, cudaStream_t stream)
{
	rconvert_kernel<uint16_t><<<(N/512)+1,512,0,stream>>>(N,in,out,conv,offset,512);
}

template<>
void rconvert<int16_t,double>(long long int N, double* in, int16_t* out,
								double conv, int16_t offset, cudaStream_t stream)
{
	rconvert_kernel<int16_t><<<(N/512)+1,512,0,stream>>>(N,in,out,conv,offset,512);
}

template<>
void rconvert<uint8_t,float>(long long int N, float* in, uint8_t* out,
								float conv, uint8_t offset, cudaStream_t stream)
{
	rconvert_kernelf<uint8_t><<<(N/512)+1,512,0,stream>>>(N,in,out,conv,offset,512);
}

template<>
void rconvert<uint16_t,float>(long long int N, float* in, uint16_t* out,
								float conv, uint16_t offset, cudaStream_t stream)
{
	rconvert_kernelf<uint16_t><<<(N/512)+1,512,0,stream>>>(N,in,out,conv,offset,512);
}

template<>
void rconvert<int16_t,float>(long long int N, float* in, int16_t* out,
								float conv, int16_t offset, cudaStream_t stream)
{
	rconvert_kernelf<int16_t><<<(N/512)+1,512,0,stream>>>(N,in,out,conv,offset,512);
}

