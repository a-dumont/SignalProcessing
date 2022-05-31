template<class DataType>
void FFT_CUDA(int n, DataType* in, DataType* out)
{
	cufftHandle plan;
	if (cufftPlan1d(&plan, n, CUFFT_Z2Z, 1) != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: Plan creation failed");
	}
	if (cufftExecZ2Z(plan, reinterpret_cast<cufftDoubleComplex*>(in), 
							reinterpret_cast<cufftDoubleComplex*>(out), 
							CUFFT_FORWARD) != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: ExecZ2Z Forward failed");
	}
	cufftDestroy(plan);
}

template<class DataType>
void fFFT_CUDA(int n, DataType* in, DataType* out)
{
	cufftHandle plan;
	if (cufftPlan1d(&plan, n, CUFFT_C2C, 1) != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: Plan creation failed");
	}
	if (cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(in), 
							reinterpret_cast<cufftComplex*>(out), 
							CUFFT_FORWARD) != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: ExecC2C Forward failed");
	}
	cufftDestroy(plan);
}


template<class DataType>
void FFT_Block_CUDA(long long int n, long long int size, DataType* in, DataType* out)
{

	int rank = 1; // Dimensionality
	long long int batch = n/size; // How many dft to compute
	long long int length[] = {size}; // Length of dft each dimension
	long long int inembed[] = {size*batch}; // Length of input dimensions 
	long long int onembed[] = {size*batch}; // Length of output dimensions 
	long long int idist = size; // Distance between dfts in
	long long int odist = size; // Distance between dfts out
	long long int stride = 1; // Stride
	size_t worksize[1]; // Number of gpus

	cufftHandle plan;
	if(cufftCreate(&plan) != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: Plan creation failed");
	}

	if (cufftMakePlanMany64(plan, rank, length, inembed,
        stride, idist, onembed, stride,
        odist, CUFFT_Z2Z, batch, worksize) != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: Plan creation failed");	
	}
	if (cufftExecZ2Z(plan, reinterpret_cast<cufftDoubleComplex*>(in), 
							reinterpret_cast<cufftDoubleComplex*>(out), 
							CUFFT_FORWARD) != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: ExecZ2Z Forward failed");
	}
	cufftDestroy(plan);
}

template<class DataType>
void fFFT_Block_CUDA(long long int n, long long int size, DataType* in, DataType* out)
{

	int rank = 1; // Dimensionality
	long long int batch = n/size; // How many dft to compute
	long long int length[] = {size}; // Length of dft each dimension
	long long int inembed[] = {size*batch}; // Length of input dimensions 
	long long int onembed[] = {size*batch}; // Length of output dimensions 
	long long int idist = size; // Distance between dfts in
	long long int odist = size; // Distance between dfts out
	long long int stride = 1; // Stride
	size_t worksize[1]; // Number of gpus

	cufftHandle plan;
	if(cufftCreate(&plan) != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: Plan creation failed");
	}

	if (cufftMakePlanMany64(plan, rank, length, inembed,
        stride, idist, onembed, stride,
        odist, CUFFT_C2C, batch, worksize) != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: Plan creation failed");	
	}
	if (cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(in), 
							reinterpret_cast<cufftComplex*>(out), 
							CUFFT_FORWARD) != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: ExecC2C Forward failed");
	}
	cufftDestroy(plan);
}


template<class DataType>
void iFFT_CUDA(int n, DataType* in, DataType* out)
{
	cufftHandle plan;
	if (cufftPlan1d(&plan, n, CUFFT_Z2Z, 1) != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: Plan creation failed");
	}
	if (cufftExecZ2Z(plan, reinterpret_cast<cufftDoubleComplex*>(in), 
							reinterpret_cast<cufftDoubleComplex*>(out), 
							CUFFT_INVERSE) != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: ExecZ2Z Inverse failed");
	}
	cufftDestroy(plan);
}

template<class DataType>
void ifFFT_CUDA(int n, DataType* in, DataType* out)
{
	cufftHandle plan;
	if (cufftPlan1d(&plan, n, CUFFT_C2C, 1) != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: Plan creation failed");
	}
	if (cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(in), 
							reinterpret_cast<cufftComplex*>(out), 
							CUFFT_INVERSE) != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: ExecC2C Inverse failed");
	}
	cufftDestroy(plan);
}


template<class DataTypeIn, class DataTypeOut>
void rFFT_CUDA(int n, DataTypeIn* in, DataTypeOut* out)
{
	cufftHandle plan;
	if (cufftPlan1d(&plan, n, CUFFT_D2Z, 1) != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: Plan creation failed");
	}
	if (cufftExecD2Z(plan, reinterpret_cast<cufftDoubleReal*>(in), 
							reinterpret_cast<cufftDoubleComplex*>(out)) != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: ExecD2Z failed");
	}
	cufftDestroy(plan);
}

template<class DataTypeIn, class DataTypeOut>
void rfFFT_CUDA(int n, DataTypeIn* in, DataTypeOut* out)
{
	cufftHandle plan;
	if (cufftPlan1d(&plan, n, CUFFT_R2C, 1) != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: Plan creation failed");
	}
	if (cufftExecR2C(plan, reinterpret_cast<cufftReal*>(in), 
							reinterpret_cast<cufftComplex*>(out)) != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: ExecR2C failed");
	}
	cufftDestroy(plan);
}


template<class DataTypeIn, class DataTypeOut>
void rFFT_Block_CUDA(int n, int size, DataTypeIn* in, DataTypeOut* out)
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
		throw std::runtime_error("CUFFT error: Plan creation failed");	
	}
	if (cufftExecD2Z(plan, reinterpret_cast<cufftDoubleReal*>(in), 
							reinterpret_cast<cufftDoubleComplex*>(out)) != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: ExecD2Z failed");
	}
	cufftDestroy(plan);
}

template<class DataTypeIn, class DataTypeOut>
void rfFFT_Block_CUDA(int n, int size, DataTypeIn* in, DataTypeOut* out)
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
		throw std::runtime_error("CUFFT error: Plan creation failed");	
	}
	if (cufftExecR2C(plan, reinterpret_cast<cufftReal*>(in), 
							reinterpret_cast<cufftComplex*>(out)) != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: ExecR2C failed");
	}
	cufftDestroy(plan);
}


template<class DataTypeIn, class DataTypeOut>
void irFFT_CUDA(int n, DataTypeIn* in, DataTypeOut* out)
{
	cufftHandle plan;
	if (cufftPlan1d(&plan, n, CUFFT_Z2D, 1) != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: Plan creation failed");
	}
	if (cufftExecZ2D(plan, reinterpret_cast<cufftDoubleComplex*>(in), 
							reinterpret_cast<cufftDoubleReal*>(out)) != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: ExecZ2D failed");
	}
	cufftDestroy(plan);
}

template<class DataTypeIn, class DataTypeOut>
void irfFFT_CUDA(int n, DataTypeIn* in, DataTypeOut* out)
{
	cufftHandle plan;
	if (cufftPlan1d(&plan, n, CUFFT_C2R, 1) != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: Plan creation failed");
	}
	if (cufftExecC2R(plan, reinterpret_cast<cufftComplex*>(in), 
							reinterpret_cast<cufftReal*>(out)) != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: ExecC2R failed");
	}
	cufftDestroy(plan);
}


