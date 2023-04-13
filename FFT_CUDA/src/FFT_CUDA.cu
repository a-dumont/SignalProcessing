#include "FFT_CUDA_cu.h"

template<class DataType>
__global__ void convertDoubleComplex_kernel(long long int N, DataType* in, cuDoubleComplex* out, 
				double conv, DataType offset, int threads)
{
	int i = threadIdx.x+blockIdx.x*threads;
	if(i<N)
	{
		out[i] = make_cuDoubleComplex(conv*in[i]-conv*offset,0);
	}
}

template<class DataType>
__global__ void convertFloatComplex_kernel(long long int N, DataType* in, cuFloatComplex* out, 
				float conv, DataType offset, int threads)
{
	int i = threadIdx.x+blockIdx.x*threads;
	if(i<N)
	{
		out[i] = make_cuFloatComplex(conv*in[i]-conv*offset,0);
	}
}

template<class DataType, class DataType2>
void convertComplex(long long int N, DataType* in, std::complex<DataType2>* out, 
				DataType2 conv, DataType offset, cudaStream_t stream){}

template<>
void convertComplex<uint8_t,double>(long long int N, uint8_t* in, std::complex<double>* out, 
				double conv, uint8_t offset, cudaStream_t stream)
{
	convertDoubleComplex_kernel<uint8_t><<<(N/512)+1,512,0,stream>>>(
					N,in,reinterpret_cast<cuDoubleComplex*>(out),conv,offset,512);
	cudaDeviceSynchronize();
}

template<>
void convertComplex<uint16_t,double>(long long int N, uint16_t* in, std::complex<double>* out, 
				double conv, uint16_t offset, cudaStream_t stream)
{
	convertDoubleComplex_kernel<uint16_t><<<(N/512)+1,512,0,stream>>>(
					N,in,reinterpret_cast<cuDoubleComplex*>(out),conv,offset,512);
	cudaDeviceSynchronize();
}

template<>
void convertComplex<int16_t,double>(long long int N, int16_t* in, std::complex<double>* out, 
				double conv, int16_t offset, cudaStream_t stream)
{
	convertDoubleComplex_kernel<int16_t><<<(N/512)+1,512,0,stream>>>(
					N,in,reinterpret_cast<cuDoubleComplex*>(out),conv,offset,512);
	cudaDeviceSynchronize();
}

template<>
void convertComplex<uint8_t,float>(long long int N, uint8_t* in, std::complex<float>* out, 
				float conv, uint8_t offset, cudaStream_t stream)
{
	convertFloatComplex_kernel<uint8_t><<<(N/512)+1,512,0,stream>>>(
					N,in,reinterpret_cast<cuFloatComplex*>(out),conv,offset,512);
	cudaDeviceSynchronize();
}

template<>
void convertComplex<uint16_t,float>(long long int N, uint16_t* in, std::complex<float>* out, 
				float conv, uint16_t offset, cudaStream_t stream)
{
	convertFloatComplex_kernel<uint16_t><<<(N/512)+1,512,0,stream>>>(
					N,in,reinterpret_cast<cuFloatComplex*>(out),conv,offset,512);
	cudaDeviceSynchronize();
}

template<>
void convertComplex<int16_t,float>(long long int N, int16_t* in, std::complex<float>* out, 
				float conv, int16_t offset, cudaStream_t stream)
{
	convertFloatComplex_kernel<int16_t><<<(N/512)+1,512,0,stream>>>(
					N,in,reinterpret_cast<cuFloatComplex*>(out),conv,offset,512);
	cudaDeviceSynchronize();
}


template<class DataType>
__global__ void convertDouble_kernel(long long int N, DataType* in, double* out, 
				double conv, DataType offset, int threads)
{
	int i = threadIdx.x+blockIdx.x*threads;
	if(i<N)
	{
		out[i] = conv*in[i]-conv*offset;
	}
}

template<class DataType>
__global__ void convertFloat_kernel(long long int N, DataType* in, float* out, 
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
	convertDouble_kernel<uint8_t><<<(N/512)+1,512,0,stream>>>(N,in,out,conv,offset,512);
	cudaDeviceSynchronize();
}

template<>
void convert<uint16_t,double>(long long int N, uint16_t* in, double* out, 
				double conv, uint16_t offset, cudaStream_t stream)
{
	convertDouble_kernel<uint16_t><<<(N/512)+1,512,0,stream>>>(N,in,out,conv,offset,512);
	cudaDeviceSynchronize();
}

template<>
void convert<uint8_t,float>(long long int N, uint8_t* in, float* out, 
				float conv, uint8_t offset, cudaStream_t stream)
{
	convertFloat_kernel<uint8_t><<<(N/512)+1,512,0,stream>>>(N,in,out,conv,offset,512);
	cudaDeviceSynchronize();
}

template<>
void convert<uint16_t,float>(long long int N, uint16_t* in, float* out, 
				float conv, uint16_t offset, cudaStream_t stream)
{
	convertFloat_kernel<uint16_t><<<(N/512)+1,512,0,stream>>>(N,in,out,conv,offset,512);
	cudaDeviceSynchronize();
}
