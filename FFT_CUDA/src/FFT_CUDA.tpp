// Fullsize FFT
template<class DataType>
void FFT_CUDA(int n, DataType* in){}

template<>
void FFT_CUDA<dbl_complex>(int n, dbl_complex* in)
{
	cufftHandle plan;
	if (cufftPlan1d(&plan, n, CUFFT_Z2Z, 1) != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: Plan creation failed");
	}
	if (cufftExecZ2Z(plan, reinterpret_cast<cufftDoubleComplex*>(in), 
							reinterpret_cast<cufftDoubleComplex*>(in), 
							CUFFT_FORWARD) != CUFFT_SUCCESS)
	{
		cufftDestroy(plan);
		throw std::runtime_error("CUFFT error: ExecZ2Z Forward failed");
	}
	cufftDestroy(plan);
}

template<>
void FFT_CUDA<flt_complex>(int n, flt_complex* in)
{
	cufftHandle plan;
	if (cufftPlan1d(&plan, n, CUFFT_C2C, 1) != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: Plan creation failed");
	}
	if (cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(in), 
							reinterpret_cast<cufftComplex*>(in), 
							CUFFT_FORWARD) != CUFFT_SUCCESS)
	{
		cufftDestroy(plan);
		throw std::runtime_error("CUFFT error: ExecC2C Forward failed");
	}
	cufftDestroy(plan);
}

// Batches of small FFTs
template<class DataType, class DataType2>
void FFT_Block_CUDA(DataType2 n, DataType2 size, DataType* in){}

template<>
void FFT_Block_CUDA<dbl_complex, long long int>(
				long long int n, long long int size, dbl_complex* in)
{
	int rank = 1; // Dimensionality
	long long int batch = n/size; // How many dft to compute
	long long int length[] = {size}; // Length of dft each dimension
	long long int inembed[] = {size*batch}; // Length of input dimensions 
	long long int idist = size; // Distance between dfts in
	long long int stride = 1; // Stride
	size_t worksize[1]; // Number of gpus

	cufftHandle plan;
	if(cufftCreate(&plan) != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: Plan creation failed");
	}

	if (cufftMakePlanMany64(plan, rank, length, inembed,
        stride, idist, inembed, stride,
        idist, CUFFT_Z2Z, batch, worksize) != CUFFT_SUCCESS)
	{
		cufftDestroy(plan);
		throw std::runtime_error("CUFFT error: Plan creation failed");	
	}
	if (cufftExecZ2Z(plan, reinterpret_cast<cufftDoubleComplex*>(in), 
							reinterpret_cast<cufftDoubleComplex*>(in), 
							CUFFT_FORWARD) != CUFFT_SUCCESS)
	{
		cufftDestroy(plan);
		throw std::runtime_error("CUFFT error: ExecZ2Z Forward failed");
	}
	cufftDestroy(plan);
}

template<>
void FFT_Block_CUDA<flt_complex, long long int>(
				long long int n, long long int size, flt_complex* in)
{
	int rank = 1; // Dimensionality
	long long int batch = n/size; // How many dft to compute
	long long int length[] = {size}; // Length of dft each dimension
	long long int inembed[] = {size*batch}; // Length of input dimensions 
	long long int idist = size; // Distance between dfts in
	long long int stride = 1; // Stride
	size_t worksize[1]; // Number of gpus

	cufftHandle plan;
	if(cufftCreate(&plan) != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: Plan creation failed");
	}

	if (cufftMakePlanMany64(plan, rank, length, inembed,
        stride, idist, inembed, stride,
        idist, CUFFT_C2C, batch, worksize) != CUFFT_SUCCESS)
	{
		cufftDestroy(plan);
		throw std::runtime_error("CUFFT error: Plan creation failed");	
	}
	if (cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(in), 
							reinterpret_cast<cufftComplex*>(in), 
							CUFFT_FORWARD) != CUFFT_SUCCESS)
	{
		cufftDestroy(plan);
		throw std::runtime_error("CUFFT error: ExecC2C Forward failed");
	}
	cufftDestroy(plan);
}

// Async version of small FFTs
template<class DataType, class DataType2>
void makePlan(cufftHandle* plan, DataType2 size, DataType2 batch){}

template<>
void  makePlan<dbl_complex, long long int>(cufftHandle* plan, 
				long long int size, long long int batch)
{
	cufftResult_t res = cufftPlan1d(plan, size, CUFFT_Z2Z, batch);
	if (res != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: Plan creation Failed");
	}
}

template<>
void  makePlan<flt_complex, long long int>(cufftHandle* plan,
				long long int size, long long int batch)
{
	cufftResult_t res = cufftPlan1d(plan, size, CUFFT_C2C , batch);
	if (res != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: Plan creation Failed");
	}
}

template<class DataType>
void FFT_Block_Async_CUDA(DataType* in, cufftHandle plan, cudaStream_t stream){}

template<>
void FFT_Block_Async_CUDA<dbl_complex>(dbl_complex* in, cufftHandle plan, cudaStream_t stream)
{
	cufftSetStream(plan, stream);	
	if (cufftExecZ2Z(plan, reinterpret_cast<cufftDoubleComplex*>(in), 
							reinterpret_cast<cufftDoubleComplex*>(in), 
							CUFFT_FORWARD) != CUFFT_SUCCESS)
	{
		cufftDestroy(plan);
		throw std::runtime_error("CUFFT error: ExecZ2Z Forward failed");
	}
}

template<>
void FFT_Block_Async_CUDA<flt_complex>(flt_complex* in, cufftHandle plan, cudaStream_t stream)
{
	cufftSetStream(plan, stream);	
	if (cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(in), 
							reinterpret_cast<cufftComplex*>(in), 
							CUFFT_FORWARD) != CUFFT_SUCCESS)
	{
		cufftDestroy(plan);
		throw std::runtime_error("CUFFT error: ExecC2C Forward failed");
	}
}

// Inverse FFT

template<class DataType>
void iFFT_CUDA(int n, DataType* in){}

template<>
void iFFT_CUDA<dbl_complex>(int n, dbl_complex* in)
{
	cufftHandle plan;
	if (cufftPlan1d(&plan, n, CUFFT_Z2Z, 1) != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: Plan creation failed");
	}
	if (cufftExecZ2Z(plan, reinterpret_cast<cufftDoubleComplex*>(in), 
							reinterpret_cast<cufftDoubleComplex*>(in), 
							CUFFT_INVERSE) != CUFFT_SUCCESS)
	{
		cufftDestroy(plan);
		throw std::runtime_error("CUFFT error: ExecZ2Z Inverse failed");
	}
	cufftDestroy(plan);
}

template<>
void iFFT_CUDA<flt_complex>(int n, flt_complex* in)
{
	cufftHandle plan;
	if (cufftPlan1d(&plan, n, CUFFT_C2C, 1) != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: Plan creation failed");
	}
	if (cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(in), 
							reinterpret_cast<cufftComplex*>(in), 
							CUFFT_INVERSE) != CUFFT_SUCCESS)
	{
		cufftDestroy(plan);
		throw std::runtime_error("CUFFT error: ExecC2C Inverse failed");
	}
	cufftDestroy(plan);
}

//Fullsize rFFT
template<class DataType>
void rFFT_CUDA(int n, DataType* in){}

template<>
void rFFT_CUDA<std::complex<double>>(int n, std::complex<double>* in)
{
	cufftHandle plan;
	if (cufftPlan1d(&plan, n, CUFFT_D2Z, 1) != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: Plan creation failed");
	}
	if (cufftExecD2Z(plan, reinterpret_cast<cufftDoubleReal*>(in), 
							reinterpret_cast<cufftDoubleComplex*>(in)) != CUFFT_SUCCESS)
	{
		cufftDestroy(plan);
		throw std::runtime_error("CUFFT error: ExecD2Z failed");
	}
	cufftDestroy(plan);
}

template<>
void rFFT_CUDA<std::complex<float>>(int n, std::complex<float>* in)
{
	cufftHandle plan;
	if (cufftPlan1d(&plan, n, CUFFT_R2C, 1) != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: Plan creation failed");
	}
	if (cufftExecR2C(plan, reinterpret_cast<cufftReal*>(in), 
							reinterpret_cast<cufftComplex*>(in)) != CUFFT_SUCCESS)
	{
		cufftDestroy(plan);
		throw std::runtime_error("CUFFT error: ExecR2C failed");
	}
	cufftDestroy(plan);
}

//Small rFFTs
template <class DataType>
void rFFT_Block_CUDA(int n, int size, DataType* in){}

template<>
void rFFT_Block_CUDA<std::complex<double>>(int n, int size, std::complex<double>* in)
{

	int rank = 1; // Dimensionality
	long long int batch = n/size; // How many dft to compute
	long long int length[] = {size}; // Length of each dimension
	long long int inembed[] = {size*batch}; // Length of input dimensions 
	long long int onembed[] = {(size/2+1)*batch}; // Length of output dimensions 
	long long int idist = size; // Distance between dfts in
	long long int odist = size/2+1; // Distance between dfts out
	long long int stride = 1; // Stride
	size_t worksize[1]; // Number of gpus

	cufftHandle plan;
	if(cufftCreate(&plan) != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: Plan creation failed");
	}

	if (cufftMakePlanMany64(plan, rank, length, inembed,
        stride, idist, onembed, stride,
        odist, CUFFT_D2Z, batch, worksize) != CUFFT_SUCCESS)
	{
		cufftDestroy(plan);
		throw std::runtime_error("CUFFT error: Plan initialization failed");
	}
	if (cufftExecD2Z(plan, reinterpret_cast<cufftDoubleReal*>(in), 
							reinterpret_cast<cufftDoubleComplex*>(in)) != CUFFT_SUCCESS)
	{
		cufftDestroy(plan);
		throw std::runtime_error("CUFFT error: ExecD2Z failed");
	}
	cufftDestroy(plan);
}

template<>
void rFFT_Block_CUDA<std::complex<float>>(int n, int size, std::complex<float>* in)
{

	int rank = 1; // Dimensionality
	long long int batch = n/size; // How many dft to compute
	long long int length[] = {size}; // Length of each dimension
	long long int inembed[] = {size*batch}; // Length of input dimensions 
	long long int onembed[] = {(size/2+1)*batch}; // Length of output dimensions 
	long long int idist = size; // Distance between dfts in
	long long int odist = size/2+1; // Distance between dfts out
	long long int stride = 1; // Stride
	size_t worksize[1]; // Number of gpus

	cufftHandle plan;
	if(cufftCreate(&plan) != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: Plan creation failed");
	}

	if (cufftMakePlanMany64(plan, rank, length, inembed,
        stride, idist, onembed, stride,
        odist, CUFFT_R2C, batch, worksize) != CUFFT_SUCCESS)
	{
		cufftDestroy(plan);
		throw std::runtime_error("CUFFT error: Plan initialization failed");
	}
	if (cufftExecR2C(plan, reinterpret_cast<cufftReal*>(in), 
							reinterpret_cast<cufftComplex*>(in)) != CUFFT_SUCCESS)
	{
		cufftDestroy(plan);
		throw std::runtime_error("CUFFT error: ExecR2C failed");
	}
	cufftDestroy(plan);
}

//Async rFFTs
template<>
void  makePlan<double, long long int>(cufftHandle* plan, 
				long long int size, long long int batch)
{
	cufftResult_t res = cufftPlan1d(plan, size, CUFFT_D2Z, batch);
	if (res != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: Plan creation Failed");
	}
}

template<>
void  makePlan<float, long long int>(cufftHandle* plan,
				long long int size, long long int batch)
{
	cufftResult_t res = cufftPlan1d(plan, size, CUFFT_R2C , batch);
	if (res != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: Plan creation Failed");
	}
}

template<class DataType>
void rFFT_Block_Async_CUDA(DataType* in, cufftHandle plan, cudaStream_t stream){}

template<>
void rFFT_Block_Async_CUDA<std::complex<double>>(std::complex<double>* in, cufftHandle plan, cudaStream_t stream)
{
	cufftSetStream(plan, stream);	
	if (cufftExecD2Z(plan, reinterpret_cast<cufftDoubleReal*>(in), 
							reinterpret_cast<cufftDoubleComplex*>(in)) != CUFFT_SUCCESS)
	{
		cufftDestroy(plan);
		throw std::runtime_error("CUFFT error: ExecD2Z Forward failed");
	}
}

template<>
void rFFT_Block_Async_CUDA<std::complex<float>>(std::complex<float>* in, 
				cufftHandle plan, cudaStream_t stream)
{
	cufftSetStream(plan, stream);	
	if (cufftExecR2C(plan, reinterpret_cast<cufftReal*>(in), 
							reinterpret_cast<cufftComplex*>(in)) != CUFFT_SUCCESS)
	{
		cufftDestroy(plan);
		throw std::runtime_error("CUFFT error: ExecR2C Forward failed");
	}
}


//Fullsize irFFT
template<class DataType>
void irFFT_CUDA(int n, DataType* in){}

template<>
void irFFT_CUDA<double>(int n, double* in)
{
	cufftHandle plan;
	if (cufftPlan1d(&plan, n, CUFFT_Z2D, 1) != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: Plan creation failed");
	}
	if (cufftExecZ2D(plan, reinterpret_cast<cufftDoubleComplex*>(in), 
							reinterpret_cast<cufftDoubleReal*>(in)) != CUFFT_SUCCESS)
	{
		cufftDestroy(plan);
		throw std::runtime_error("CUFFT error: ExecZ2D failed");
	}
	cufftDestroy(plan);
}

template<>
void irFFT_CUDA<float>(int n, float* in)
{
	cufftHandle plan;
	if (cufftPlan1d(&plan, n, CUFFT_C2R, 1) != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: Plan creation failed");
	}
	if (cufftExecC2R(plan, reinterpret_cast<cufftComplex*>(in), 
							reinterpret_cast<cufftReal*>(in)) != CUFFT_SUCCESS)
	{
		cufftDestroy(plan);
		throw std::runtime_error("CUFFT error: ExecC2R failed");
	}
	cufftDestroy(plan);
}

template<class DataType>
void makePlanInv(cufftHandle* plan,
				long long int size, long long int batch){}

template<>
void makePlanInv<double>(cufftHandle* plan, 
				long long int size, long long int batch)
{
	cufftResult_t res = cufftPlan1d(plan, size, CUFFT_Z2D, batch);
	if (res != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: Plan creation Failed");
	}
}

template<>
void makePlanInv<float>(cufftHandle* plan, 
				long long int size, long long int batch)
{
	cufftResult_t res = cufftPlan1d(plan, size, CUFFT_C2R, batch);
	if (res != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: Plan creation Failed");
	}
}

template<class DataType>
void irFFT_Block_Async_CUDA(DataType* in, cufftHandle plan, cudaStream_t stream){}

template<>
void irFFT_Block_Async_CUDA<std::complex<double>>(std::complex<double>* in, cufftHandle plan, cudaStream_t stream)
{
	cufftSetStream(plan, stream);	
	if (cufftExecZ2D(plan, reinterpret_cast<cufftDoubleComplex*>(in), 
							reinterpret_cast<cufftDoubleReal*>(in)) != CUFFT_SUCCESS)
	{
		cufftDestroy(plan);
		throw std::runtime_error("CUFFT error: ExecD2Z Forward failed");
	}
}

template<>
void irFFT_Block_Async_CUDA<std::complex<float>>(std::complex<float>* in, 
				cufftHandle plan, cudaStream_t stream)
{
	cufftSetStream(plan, stream);	
	if (cufftExecC2R(plan, reinterpret_cast<cufftComplex*>(in), 
							reinterpret_cast<cufftReal*>(in)) != CUFFT_SUCCESS)
	{
		cufftDestroy(plan);
		throw std::runtime_error("CUFFT error: ExecR2C Forward failed");
	}
}

/*
template<class DataType>
void filter_single(int64_t N, DataType* data, float* filter, DataType offset)
{
	float* gpu;
	cudaStream_t streams[2];
	cudaStreamCreate(&streams[0]);
	cudaStreamCreate(&streams[1]);
	cufftHandle plan;

	cudaMalloc((void**)&gpu, 3*(N/2+1)*sizeof(float));
	cudaMemcpyAsync(gpu+(N+2),data,sizeof(DataType)*N,cudaMemcpyHostToDevice,streams[0]);
	convert<DataType,float>(N,reinterpret_cast<DataType*>(gpu+N+2),gpu,1.0,offset,streams[0]);
	cudaMemcpyAsync(gpu+(N+2),filter,sizeof(float)*N/2+1,cudaMemcpyHostToDevice,streams[1]);

	if (cufftPlan1d(&plan, N, CUFFT_R2C, 1) != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: Plan creation failed");
	}
	if (cufftExecR2C(plan, reinterpret_cast<cufftReal*>(gpu),
							reinterpret_cast<cufftComplex*>(gpu)) != CUFFT_SUCCESS)
	{
		cufftDestroy(plan);
		throw std::runtime_error("CUFFT error: ExecR2C failed");
	}

	filter_CUDA<float>(N/2+1,gpu,gpu+(N+2),streams[0]);

	if (cufftPlan1d(&plan, N, CUFFT_C2R, 1) != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: Plan creation failed");
	}
	if (cufftExecC2R(plan, reinterpret_cast<cufftComplex*>(gpu),
							reinterpret_cast<cufftReal*>(gpu)) != CUFFT_SUCCESS)
	{
		cufftDestroy(plan);
		throw std::runtime_error("CUFFT error: ExecC2R failed");
	}

	rconvert<DataType,float>(N,gpu,reinterpret_cast<DataType*>(gpu),1.0/N,offset,streams[0]);
	cudaMemcpyAsync(data,gpu,sizeof(DataType)*N,cudaMemcpyDeviceToHost,streams[0]);

	cufftDestroy(plan);
	cudaStreamDestroy(streams[0]);
	cudaStreamDestroy(streams[1]);
	cudaFree(gpu);
}

template<class DataType>
void filter_dual(int64_t N, DataType* data1, DataType* data2,
				float* filter1, float* filter2, DataType offset)
{
	float* gpu;
	cudaStream_t streams[2];
	cudaStreamCreate(&streams[0]);
	cudaStreamCreate(&streams[1]);
	cufftHandle plan1, plan2;
	cufftSetStream(plan1, streams[0]);
	cufftSetStream(plan2, streams[1]);

	cudaMalloc((void**)&gpu, 6*(N/2+1)*sizeof(float));
	cudaMemcpyAsync(gpu+N+2,data1,sizeof(DataType)*N,cudaMemcpyHostToDevice,streams[0]);
	convert(N,reinterpret_cast<DataType*>(gpu+N+2),gpu,1.0,offset,streams[0]);
	cudaMemcpyAsync(gpu+(N+2),filter1,sizeof(float)*N/2+1,cudaMemcpyHostToDevice,streams[1]);

	cudaMemcpyAsync(gpu+(5*N/2+3),data2,sizeof(DataType)*N,cudaMemcpyHostToDevice,streams[0]);
	convert(N,reinterpret_cast<DataType*>(gpu+(5*N/2+3)),gpu+(3*N/2+3),1.0,offset,streams[0]);
	cudaMemcpyAsync(gpu+(5*N/2+5),filter2,sizeof(float)*N/2+1,cudaMemcpyHostToDevice,streams[1]);

	if (cufftPlan1d(&plan1, N, CUFFT_R2C, 1) != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: Plan creation failed");
	}
	if (cufftExecR2C(plan1, reinterpret_cast<cufftReal*>(gpu),
							reinterpret_cast<cufftComplex*>(gpu)) != CUFFT_SUCCESS)
	{
		cufftDestroy(plan1);
		throw std::runtime_error("CUFFT error: ExecR2C failed");
	}
	if (cufftPlan1d(&plan2, N, CUFFT_R2C, 1) != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: Plan creation failed");
	}
	if (cufftExecR2C(plan2, reinterpret_cast<cufftReal*>(gpu+(3*N/2+3)),
							reinterpret_cast<cufftComplex*>(gpu+(3*N/2+3))) != CUFFT_SUCCESS)
	{
		cufftDestroy(plan2);
		throw std::runtime_error("CUFFT error: ExecR2C failed");
	}

	filter_CUDA<float>(N/2+1,gpu,gpu+N+2,streams[0]);
	filter_CUDA<float>(N/2+1,gpu+3*N/2+3,gpu+5*N/2+5,streams[1]);

	if (cufftPlan1d(&plan1, N, CUFFT_C2R, 1) != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: Plan creation failed");
	}
	if (cufftExecC2R(plan1, reinterpret_cast<cufftComplex*>(gpu),
							reinterpret_cast<cufftReal*>(gpu)) != CUFFT_SUCCESS)
	{
		cufftDestroy(plan1);
		throw std::runtime_error("CUFFT error: ExecC2R failed");
	}
	if (cufftPlan1d(&plan2, N, CUFFT_C2R, 1) != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: Plan creation failed");
	}
	if (cufftExecC2R(plan2, reinterpret_cast<cufftComplex*>(gpu+(3*N/2+3)),
							reinterpret_cast<cufftReal*>(gpu+(3*N/2+3))) != CUFFT_SUCCESS)
	{
		cufftDestroy(plan2);
		throw std::runtime_error("CUFFT error: ExecC2R failed");
	}


	rconvert(N,gpu,reinterpret_cast<DataType*>(gpu),1.0/N,offset,streams[0]);
	cudaMemcpyAsync(data1,gpu,sizeof(DataType)*N,cudaMemcpyDeviceToHost,streams[0]);

	rconvert(N,gpu+(3*N/2+3),reinterpret_cast<DataType*>(gpu+(3*N/2+3)),1.0/N,offset,streams[1]);
	cudaMemcpyAsync(data2,gpu+(3*N/2+3),sizeof(DataType)*N,cudaMemcpyDeviceToHost,streams[0]);

	cufftDestroy(plan1);
	cufftDestroy(plan2);
	cudaStreamDestroy(streams[0]);
	cudaStreamDestroy(streams[1]);
	cudaFree(gpu);
}
*/
