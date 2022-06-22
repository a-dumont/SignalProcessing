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
		throw std::runtime_error("CUFFT error: Plan creation failed");	
	}
	if (cufftExecZ2Z(plan, reinterpret_cast<cufftDoubleComplex*>(in), 
							reinterpret_cast<cufftDoubleComplex*>(in), 
							CUFFT_FORWARD) != CUFFT_SUCCESS)
	{
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
		throw std::runtime_error("CUFFT error: Plan creation failed");	
	}
	if (cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(in), 
							reinterpret_cast<cufftComplex*>(in), 
							CUFFT_FORWARD) != CUFFT_SUCCESS)
	{
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
		throw std::runtime_error("CUFFT error: Plan initialization failed");
		cufftDestroy(plan);
	}
	if (cufftExecD2Z(plan, reinterpret_cast<cufftDoubleReal*>(in), 
							reinterpret_cast<cufftDoubleComplex*>(in)) != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: ExecD2Z failed");
		cufftDestroy(plan);
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
		throw std::runtime_error("CUFFT error: Plan initialization failed");
		cufftDestroy(plan);
	}
	if (cufftExecR2C(plan, reinterpret_cast<cufftReal*>(in), 
							reinterpret_cast<cufftComplex*>(in)) != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: ExecR2C failed");
		cufftDestroy(plan);
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
		throw std::runtime_error("CUFFT error: ExecC2R failed");
	}
	cufftDestroy(plan);
}
