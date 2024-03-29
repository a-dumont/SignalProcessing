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
}

template<>
void autocorrelation_cuda<float>(long long int N, std::complex<float>* in, float* out)
{
	int threads = 512;
	long long int blocks = N/threads;
	autocorrelation_cuda_kernelf<<<blocks+1,threads>>>(N,
					reinterpret_cast<cuFloatComplex*>(in),out,threads);
}

__global__	void autocorrelation_cuda_kernel2(long long int N, double* in, 
				double* out, int threads)
{
	// Compute the correlation
	long long int i = threadIdx.x+blockIdx.x*threads;
	if(i<N)
	{
		out[2*i] = in[2*i]*in[2*i]+in[2*i+1]*in[2*i+1];
		out[2*i+1] = 0;
	}
}

__global__	void autocorrelation_cuda_kernel2f(long long int N, float* in, 
				float* out, int threads)
{
	// Compute the correlation
	long long int i = threadIdx.x+blockIdx.x*threads;
	if(i<N)
	{
		out[2*i] = in[2*i]*in[2*i]+in[2*i+1]*in[2*i+1];
		out[2*i+1] = 0;
	}
}

template<class DataType>
void autocorrelation_cuda2(long long int N, DataType* in, DataType* out){}

template<>
void autocorrelation_cuda2<double>(long long int N, double* in, double* out)
{
	int threads = 512;
	long long int blocks = N/threads;
	autocorrelation_cuda_kernel2<<<blocks+1,threads>>>(N,in,out,threads);
}

template<>
void autocorrelation_cuda2<float>(long long int N, float* in, float* out)
{
	int threads = 512;
	long long int blocks = N/threads;
	autocorrelation_cuda_kernel2f<<<blocks+1,threads>>>(N,in,out,threads);
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
}

__global__ void reduction_kernel(long long int N, double* in, int threads)
{
	int i = threadIdx.x+blockIdx.x*threads;
	if(i<N)
	{
		in[i] += in[i+N];
	}
}

__global__ void reduction_kernelf(long long int N, float* in, int threads)
{
	int i = threadIdx.x+blockIdx.x*threads;
	if(i<N)
	{
		in[i] += in[i+N];
	}
}

__global__ void reduction_kernelll(long long int N, long long int* in, int threads)
{
	int i = threadIdx.x+blockIdx.x*threads;
	if(i<N)
	{
		in[i] += in[i+N];
	}
}


template<class DataType> 
void reduction(long long int N, DataType* in, long long int size){}

template<>
void reduction<double>(long long int N, double* in, long long int size)
{
	if (N/2 >= size)
	{
		reduction_kernel<<<N/1024+1,512>>>(N/2,in,512);
		reduction(N/2,in,size);
	}
}

template<>
void reduction<float>(long long int N, float* in, long long int size)
{
	if (N/2 >= size)
	{
		reduction_kernelf<<<N/1024+1,512>>>(N/2,in,512);
		reduction(N/2,in,size);
	}
}

template<>
void reduction<long long int>(long long int N, long long int* in, long long int size)
{
	if (N/2 >= size)
	{
		reduction_kernelll<<<N/1024+1,512>>>(N/2,in,512);
		reduction(N/2,in,size);
	}
}

__global__ void reduction_general_kernel(long long int N, double* in, long long int size, 
				int threads)
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

__global__ void reduction_general_kernelf(long long int N, float* in, long long int size, 
				int threads)
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
void reduction_general(long long int N, DataType* in, long long int size){}

template<>
void reduction_general<double>(long long int N, double* in, long long int size)
{
	int power = std::log2(N/size);
	long long int n = size*1<<power;
	if(n < N)
	{
		long long int diff = N-n+size;
		reduction_general_kernel<<<diff/512+1,512>>>(diff,in+n-size,size,512);
	}
	reduction(n,in,size);
}

template<>
void reduction_general<float>(long long int N, float* in, long long int size)
{
	int power = std::log2(N/size);
	long long int n = size*1<<power;
	if(n < N)
	{
		long long int diff = N-n+size;
		reduction_general_kernelf<<<diff/512+1,512>>>(diff,in+n-size,size,512);
	}
	reduction(n,in,size);
}

template<class DataType>
__global__ void convertComplex_kernel(long long int N, DataType* in, cuDoubleComplex* out,
				double conv, DataType offset, int threads)
{
	int i = threadIdx.x+blockIdx.x*threads;
	if(i<N)
	{
		out[i] = make_cuDoubleComplex(conv*in[i]-conv*offset,0);
	}
}

template<class DataType>
__global__ void convertComplex_kernelf(long long int N, DataType* in, cuFloatComplex* out,
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
	convertComplex_kernel<uint8_t><<<(N/512)+1,512,0,stream>>>(
					N,in,reinterpret_cast<cuDoubleComplex*>(out),conv,offset,512);
}

template<>
void convertComplex<uint16_t,double>(long long int N, uint16_t* in, std::complex<double>* out,
				double conv, uint16_t offset, cudaStream_t stream)
{
	convertComplex_kernel<uint16_t><<<(N/512)+1,512,0,stream>>>(
					N,in,reinterpret_cast<cuDoubleComplex*>(out),conv,offset,512);
}

template<>
void convertComplex<int16_t,double>(long long int N, int16_t* in, std::complex<double>* out,
				double conv, int16_t offset, cudaStream_t stream)
{
	convertComplex_kernel<int16_t><<<(N/512)+1,512,0,stream>>>(
					N,in,reinterpret_cast<cuDoubleComplex*>(out),conv,offset,512);
}


template<>
void convertComplex<uint8_t,float>(long long int N, uint8_t* in, std::complex<float>* out,
				float conv, uint8_t offset, cudaStream_t stream)
{
	convertComplex_kernelf<uint8_t><<<(N/512)+1,512,0,stream>>>(
					N,in,reinterpret_cast<cuFloatComplex*>(out),conv,offset,512);
}

template<>
void convertComplex<uint16_t,float>(long long int N, uint16_t* in, std::complex<float>* out,
				float conv, uint16_t offset, cudaStream_t stream)
{
	convertComplex_kernelf<uint16_t><<<(N/512)+1,512,0,stream>>>(
					N,in,reinterpret_cast<cuFloatComplex*>(out),conv,offset,512);
}

template<>
void convertComplex<int16_t,float>(long long int N, int16_t* in, std::complex<float>* out,
				float conv, int16_t offset, cudaStream_t stream)
{
	convertComplex_kernelf<int16_t><<<(N/512)+1,512,0,stream>>>(
					N,in,reinterpret_cast<cuFloatComplex*>(out),conv,offset,512);
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


__global__	void autocorrelation_convert_kernel(long long int N, cuFloatComplex* in, 
				double* out, int threads)
{
	// Compute the correlation
	long long int i = threadIdx.x+blockIdx.x*threads;
	double a,b;
	if(i<N)
	{
		a = (double) cuCrealf(in[i]);
		b = (double) cuCimagf(in[i]);
		out[i] = a*a+b*b;
		//out[i] = cuCrealf(in[i])*cuCrealf(in[i])+cuCimagf(in[i])*cuCimagf(in[i]);
	}  
}

void autocorrelation_convert(long long int N, std::complex<float>* in, double* out)
{
	int threads = 512;
	long long int blocks = N/threads;
	autocorrelation_convert_kernel<<<blocks+1,threads>>>(N,
					reinterpret_cast<cuFloatComplex*>(in),out,threads);
}

__global__ void add_kernel(long long int N, double* in, double* out, int threads)
{
	long long int i = threadIdx.x+blockIdx.x*threads;
	if(i<N)
	{
		out[i] += in[i];
	}  
}

__global__ void add_kernelf(long long int N, float* in, float* out, int threads)
{
	long long int i = threadIdx.x+blockIdx.x*threads;
	if(i<N)
	{
		out[i] += in[i];
	}  
}

__global__ void add_kernelll(long long int N, long long int* in, long long int* out, int threads)
{
	long long int i = threadIdx.x+blockIdx.x*threads;
	if(i<N)
	{
		out[i] += in[i];
	}  
}


template<class DataType>
void add_cuda(long long int N, DataType* in, DataType* out){}

template<>
void add_cuda<double>(long long int N, double* in, double* out)
{
	int threads = 512;
	long long int blocks = N/threads+1;
	add_kernel<<<blocks,threads>>>(N,in,out,threads);
}

template<>
void add_cuda<float>(long long int N, float* in, float* out)
{
	int threads = 512;
	long long int blocks = N/threads+1;
	add_kernelf<<<blocks,threads>>>(N,in,out,threads);
}

template<>
void add_cuda<long long int>(long long int N, long long int* in, long long int* out)
{
	int threads = 512;
	long long int blocks = N/threads+1;
	add_kernelll<<<blocks,threads>>>(N,in,out,threads);
}


__global__	void cross_correlation_convert_kernel(long long int N, cuFloatComplex* in1, 
				cuFloatComplex* in2, double* out1, double* out2, int threads)
{
	// Compute the correlation
	int i = threadIdx.x+blockIdx.x*threads;
	cuFloatComplex temp;
	if(i<N)
	{ 
		temp = cuCmulf(in1[i],cuConjf(in2[i]));
		out1[i] = cuCrealf(temp);
		out2[i] = cuCimagf(temp);
	}
}

void crosscorrelation_convert(long long int N, std::complex<float>* in1, 
				std::complex<float>*in2, double* out1, double* out2)
{
	int threads = 512;
	long long int blocks = N/threads;
	cross_correlation_convert_kernel<<<blocks+1,threads>>>(N,
					reinterpret_cast<cuFloatComplex*>(in1),
					reinterpret_cast<cuFloatComplex*>(in2),
					out1,out2,threads);
}

__global__ void add_complex_kernel(long long int N, double* in1, double* in2, double* out, 
				int threads)
{
	long long int i = threadIdx.x+blockIdx.x*threads;
	if(i<N)
	{
		out[2*i] += in1[i];
		out[2*i+1] += in2[i];
	}  
}

void add_complex_cuda(long long int N, double* in1, double* in2, double* out)
{
	int threads = 512;
	long long int blocks = N/threads+1;
	add_complex_kernel<<<blocks,threads>>>(N,in1,in2,out,threads);
}

__global__	void complete_correlation_convert_kernel(long long int N, float* in1, 
				float* in2, double* out1, double* out2, double* out3, double* out4,int threads)
{
	// Compute the correlation
	int i = threadIdx.x+blockIdx.x*threads;
	float a,b,c,d;
	if(i<N)
	{ 
		a = in1[2*i]; b = in1[2*i+1];
		c = in2[2*i]; d = -in2[2*i+1];
		out1[i] = a*c-b*d;
		out2[i] = a*d+b*c;
		out3[i] = a*a+b*b;
		out4[i] = c*c+d*d;
	}
}

void completecorrelation_convert(long long int N, std::complex<float>* in1, 
				std::complex<float>*in2, double* out1, double* out2, double* out3, double* out4)
{
	int threads = 512;
	long long int blocks = N/threads;
	complete_correlation_convert_kernel<<<blocks+1,threads>>>(N,
					reinterpret_cast<float*>(in1),
					reinterpret_cast<float*>(in2),
					out1,out2,out3,out4,threads);
}

__global__ void round_kernel(long long int N, double* in, int threads)
{
	int i = threadIdx.x+blockIdx.x*threads;
	if(i<N)
	{ 
		in[i] = std::round(in[i]);
	}
}

__global__ void round_kernelf(long long int N, float* in, int threads)
{
	int i = threadIdx.x+blockIdx.x*threads;
	if(i<N)
	{ 
		in[i] = std::round(in[i]);
	}
}

template<class DataType>
void round(long long int N, DataType* in){}

template<>
void round<double>(long long int N, double* in)
{
	int threads = 512;
	long long int blocks = N/threads;
	round_kernel<<<blocks+1,threads>>>(N,in,threads);
}

template<>
void round<float>(long long int N, float* in)
{
	int threads = 512;
	long long int blocks = N/threads;
	round_kernelf<<<blocks+1,threads>>>(N,in,threads);
}


__global__ void llround_kernel(long long int N, double* in, long long int* out, int threads)
{
	int i = threadIdx.x+blockIdx.x*threads;
	if(i<N)
	{ 
		out[i] = std::llround(in[i]);
	}
}

__global__ void llround_kernelf(long long int N, float* in, long long int* out, int threads)
{
	int i = threadIdx.x+blockIdx.x*threads;
	if(i<N)
	{ 
		out[i] = std::llround(in[i]);
	}
}

template<class DataType>
void llround(long long int N, DataType* in, long long int* out){}

template<>
void llround<double>(long long int N, double* in, long long int* out)
{
	int threads = 512;
	long long int blocks = N/threads;
	llround_kernel<<<blocks+1,threads>>>(N,in,out,threads);
}

template<>
void llround<float>(long long int N, float* in, long long int* out)
{
	int threads = 512;
	long long int blocks = N/threads;
	llround_kernelf<<<blocks+1,threads>>>(N,in,out,threads);
}

__global__ void mul_kernel(long long int N, double* in, double m, int threads)
{
	int i = threadIdx.x+blockIdx.x*threads;
	if(i<N)
	{ 
		in[i] *= m;
	}
}

__global__ void mul_kernelf(long long int N, float* in, float m, int threads)
{
	int i = threadIdx.x+blockIdx.x*threads;
	if(i<N)
	{ 
		in[i] *= m;
	}
}

template<class DataType>
void mul(long long int N, DataType* in, DataType m){}

template<>
void mul<double>(long long int N, double* in, double m)
{
	int threads = 512;
	long long int blocks = N/threads;
	mul_kernel<<<blocks+1,threads>>>(N,in,m,threads);
}

template<>
void mul<float>(long long int N, float* in, float m)
{
	int threads = 512;
	long long int blocks = N/threads;
	mul_kernelf<<<blocks+1,threads>>>(N,in,m,threads);
}
