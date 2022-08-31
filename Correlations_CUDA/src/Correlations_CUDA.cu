#include "Correlations_CUDA.h"

__global__	void autocorrelation_cuda_kernel(long long int N, cuDoubleComplex* in, 
				double* out, int threads)
{
	// Compute the correlation
	long long int i = threadIdx.x+blockIdx.x*threads;
	if(i<N)
	{
		out[i] = cuCreal(in[i])*cuCreal(in[i])+cuCimag(in[i])*cuCimag(in[i]);
	}
}

__global__	void autocorrelation_cuda_kernelf(long long int N, cuFloatComplex* in, 
				float* out, int threads)
{
	// Compute the correlation
	long long int i = threadIdx.x+blockIdx.x*threads;
	if(i<N)
	{
		out[i] = cuCrealf(in[i])*cuCrealf(in[i])+cuCimagf(in[i])*cuCimagf(in[i]);
	}
}

template<class DataType>
void autocorrelation_cuda(long long int N, std::complex<DataType>* in, DataType* out){}

template<>
void autocorrelation_cuda<double>(long long int N, std::complex<double>* in, double* out)
{
	int threads = 512;
	long long int blocks = N/threads;
	autocorrelation_cuda_kernel<<<blocks+1,threads>>>(N,
					reinterpret_cast<cuDoubleComplex*>(in),out,threads);
	cudaDeviceSynchronize();
}

template<>
void autocorrelation_cuda<float>(long long int N, std::complex<float>* in, float* out)
{
	int threads = 512;
	long long int blocks = N/threads;
	autocorrelation_cuda_kernelf<<<blocks+1,threads>>>(N,
					reinterpret_cast<cuFloatComplex*>(in),out,threads);
	cudaDeviceSynchronize();
}

__global__	void cross_correlation_cuda_kernel(long long int N, cuDoubleComplex* in1, 
				cuDoubleComplex* in2, cuDoubleComplex* out, int threads)
{
	// Compute the correlation
	int i = threadIdx.x+blockIdx.x*threads;
	if(i<N)
	{
		out[i] = cuCmul(in1[i],cuConj(in2[i]));
	}
}

__global__	void cross_correlation_cuda_kernelf(long long int N, cuFloatComplex* in1, 
				cuFloatComplex* in2, cuFloatComplex* out, int threads)
{
	// Compute the correlation
	int i = threadIdx.x+blockIdx.x*threads;
	if(i<N)
	{
		out[i] = cuCmulf(in1[i],cuConjf(in2[i]));
	}
}

template<class DataType>
void cross_correlation_cuda(long long int N, std::complex<DataType>* in1, 
				std::complex<DataType>*in2, std::complex<DataType>* out){}

template<>
void cross_correlation_cuda<double>(long long int N, std::complex<double>* in1, 
				std::complex<double>* in2, std::complex<double>* out)
{
	int threads = 512;
	long long int blocks = N/threads;
	cross_correlation_cuda_kernel<<<blocks+1,threads>>>(N,
					reinterpret_cast<cuDoubleComplex*>(in1),
					reinterpret_cast<cuDoubleComplex*>(in2),
					reinterpret_cast<cuDoubleComplex*>(out),threads);
	cudaDeviceSynchronize();
}

template<>
void cross_correlation_cuda<float>(long long int N, std::complex<float>* in1, 
				std::complex<float>* in2, std::complex<float>* out)
{
	int threads = 512;
	long long int blocks = N/threads;
	cross_correlation_cuda_kernelf<<<blocks+1,threads>>>(N,
					reinterpret_cast<cuFloatComplex*>(in1),
					reinterpret_cast<cuFloatComplex*>(in2),
					reinterpret_cast<cuFloatComplex*>(out),threads);
	cudaDeviceSynchronize();
}

__global__	void complete_correlation_cuda_kernel(long long int N, cuDoubleComplex* in1, 
				cuDoubleComplex* in2, double* out1, double* out2, cuDoubleComplex* out3, 
				int threads)
{
	// Compute the correlation
	int i = threadIdx.x+blockIdx.x*threads;
	cuDoubleComplex a,b;
	if(i<N)
	{
		a = in1[i];
		b = in2[i];
		out1[i] = cuCreal(a)*cuCreal(a)+cuCimag(a)*cuCimag(a);
		out2[i] = cuCreal(b)*cuCreal(b)+cuCimag(b)*cuCimag(b);
		out3[i] = cuCmul(a,cuConj(b));
	}
}

__global__	void complete_correlation_cuda_kernelf(long long int N, cuFloatComplex* in1, 
				cuFloatComplex* in2, float* out1, float* out2, cuFloatComplex* out3, 
				int threads)
{
	// Compute the correlation
	int i = threadIdx.x+blockIdx.x*threads;
	cuFloatComplex a,b;
	if(i<N)
	{
		a = in1[i];
		b = in2[i];
		out1[i] = cuCrealf(a)*cuCrealf(a)+cuCimagf(a)*cuCimagf(a);
		out2[i] = cuCrealf(b)*cuCrealf(b)+cuCimagf(b)*cuCimagf(b);
		out3[i] = cuCmulf(a,cuConjf(b));
	}
}

template<class DataType>
void complete_correlation_cuda(long long int N, std::complex<DataType>* in1, 
				std::complex<DataType>*in2, 
				DataType* out1, 
				DataType* out2,
				std::complex<DataType>* out3){}

template<>
void complete_correlation_cuda<double>(long long int N, std::complex<double>* in1, 
				std::complex<double>* in2, double* out1, double* out2, std::complex<double>* out3)
{
	int threads = 512;
	long long int blocks = N/threads;
	complete_correlation_cuda_kernel<<<blocks+1,threads>>>(N,
					reinterpret_cast<cuDoubleComplex*>(in1),
					reinterpret_cast<cuDoubleComplex*>(in2),
					out1,
					out2,
					reinterpret_cast<cuDoubleComplex*>(out3),threads);
	cudaDeviceSynchronize();
}

template<>
void complete_correlation_cuda<float>(long long int N, std::complex<float>* in1, 
				std::complex<float>* in2, float* out1, float* out2, std::complex<float>* out3)
{
	int threads = 512;
	long long int blocks = N/threads;
	complete_correlation_cuda_kernelf<<<blocks+1,threads>>>(N,
					reinterpret_cast<cuFloatComplex*>(in1),
					reinterpret_cast<cuFloatComplex*>(in2),
					out1,
					out2,
					reinterpret_cast<cuFloatComplex*>(out3),threads);
	cudaDeviceSynchronize();
}

template<class DataType>
__global__ void reduction_kernel(long long int N, DataType* in, int threads)
{
	int i = threadIdx.x+blockIdx.x*threads;
	if(i<N)
	{
		in[i] += in[i+N];
	}
}

template<class DataType> 
void reduction(long long int N, DataType* in, int size)
{
	reduction_kernel<DataType><<<N/1024+1,512>>>(N/2,in,512);
	if (N/2 > size){reduction(N/2,in,size);}
}

template<class DataType>
__global__ void reduction_general_kernel(long long int N, DataType* in, int size, int threads)
{
	int i = threadIdx.x+blockIdx.x*threads;
	long long int howmany = N/size;
	if(i<size)
	{
		for(long long int j=1;j<howmany;j++)
		{
			in[i] += in[i+j*size];
		}
	}
}

template<class DataType> 
void reduction_general(long long int N, DataType* in, int size)
{
	int power = (int) std::log2(1.0*N);
	long long int n = 1<<power;
	long long int diff = N-n+size;
	reduction_general_kernel<DataType><<<diff/512+1,512>>>(diff,in[n-size],size,512);
	reduction_kernel<DataType><<<n/1024+1,512>>>(n/2,in,512);
	if (n/2 > size){reduction(n/2,in,size);}
}
